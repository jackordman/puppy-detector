#!/usr/bin/env python3
"""Puppy detector. Pulls the Wyze RTSP stream, finds dogs with YOLOv8n and
saves crops + full frames (phase 1). If a squat classifier exists it also
scores each crop and fires an event after N consecutive positives (phase 2/3).

Config comes from the environment (systemd EnvironmentFile), see config.example.
    detector.py            run forever
    detector.py probe      grab one frame, detect, print timing, save to /tmp
    detector.py selftest   exercise the pure logic without a camera
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RTSP_URL = os.environ.get("RTSP_URL", "")
OUT = Path(os.environ.get("OUT_DIR", "/srv/nas/share/puppycam"))
FPS = float(os.environ.get("FPS", "2"))
IMGSZ = int(os.environ.get("IMGSZ", "416"))
CONF = float(os.environ.get("CONF", "0.4"))
MAX_GB = float(os.environ.get("MAX_GB", "20"))
CLASSIFIER = Path(os.environ.get("CLASSIFIER", "/opt/puppy-detector/squat.pt"))
SQUAT_CONF = float(os.environ.get("SQUAT_CONF", "0.6"))
CONSECUTIVE = int(os.environ.get("CONSECUTIVE", "3"))
EVENT_COOLDOWN = float(os.environ.get("EVENT_COOLDOWN", "120"))
DOG = 16  # COCO class id


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


# ---------- frames ----------

def stream_size(url):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", url], timeout=30)
    w, h = out.decode().strip().split(",")[:2]
    return int(w), int(h)


def frames(url):
    """Yield BGR frames at FPS forever, restarting ffmpeg when the stream drops."""
    while True:
        try:
            w, h = stream_size(url)
        except Exception as e:  # noqa: BLE001
            log("probe failed:", e)
            time.sleep(10)
            continue
        n = w * h * 3
        p = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", url,
             "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, bufsize=n)
        log(f"stream {w}x{h} @ {FPS} fps")
        try:
            while True:
                buf = p.stdout.read(n)
                if len(buf) < n:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        finally:
            p.kill()
        log("stream ended, reconnecting")
        time.sleep(5)


# ---------- pure logic ----------

def pick_dog(boxes, cls, conf):
    """Largest dog box as ((x1, y1, x2, y2), score), or (None, 0)."""
    best, score, area = None, 0.0, 0
    for b, c, s in zip(boxes, cls, conf):
        if int(c) != DOG or s < CONF:
            continue
        x1, y1, x2, y2 = (int(v) for v in b)
        if (x2 - x1) * (y2 - y1) > area:
            best, score, area = (x1, y1, x2, y2), float(s), (x2 - x1) * (y2 - y1)
    return best, score


def crop(frame, box, margin=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return frame[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]


def prune(root, max_bytes):
    """Delete oldest day folders under root/crops and root/frames until under cap."""
    def size(p):
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    days = sorted({d.name for sub in ("crops", "frames") for d in (root / sub).glob("20*") if d.is_dir()})
    total = size(root / "crops") + size(root / "frames")
    removed = []
    while total > max_bytes and len(days) > 1:
        day = days.pop(0)
        for sub in ("crops", "frames"):
            d = root / sub / day
            if d.exists():
                total -= size(d)
                shutil.rmtree(d)
        removed.append(day)
    return removed


class Streak:
    """Counts consecutive positives, fires once per cooldown."""

    def __init__(self, need, cooldown):
        self.need, self.cooldown, self.n, self.last = need, cooldown, 0, float("-inf")

    def update(self, positive, now):
        self.n = self.n + 1 if positive else 0
        if self.n >= self.need and now - self.last >= self.cooldown:
            self.last = now
            self.n = 0
            return True
        return False


# ---------- main ----------

def run():
    import cv2
    from ultralytics import YOLO
    import notify

    det = YOLO("yolov8n.pt")
    cls = YOLO(str(CLASSIFIER)) if CLASSIFIER.exists() else None
    log("classifier:", CLASSIFIER if cls else "none (phase 1, collecting only)")
    streak = Streak(CONSECUTIVE, EVENT_COOLDOWN)
    last_save = last_prune = 0.0
    for frame in frames(RTSP_URL):
        now = time.time()
        r = det(frame, imgsz=IMGSZ, classes=[DOG], conf=CONF, verbose=False)[0]
        box, score = pick_dog(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
        if box is None:
            streak.update(False, now)
            continue
        c = crop(frame, box)
        ts = datetime.now()
        day, name = ts.strftime("%Y-%m-%d"), ts.strftime("%H%M%S_%f")[:-3] + ".jpg"
        (OUT / "detections").mkdir(parents=True, exist_ok=True)
        with open(OUT / "detections" / f"{day}.csv", "a", newline="") as f:
            csv.writer(f).writerow([ts.isoformat(timespec="milliseconds"), *box, round(score, 3)])
        if now - last_save >= 1.0:
            last_save = now
            for sub, img in (("crops", c), ("frames", frame)):
                d = OUT / sub / day
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if cls is not None:
            p = cls(c, imgsz=224, verbose=False)[0].probs
            positive = cls.names[p.top1] == "squat" and float(p.top1conf) >= SQUAT_CONF
            if streak.update(positive, now):
                snap = OUT / "events" / f"{ts:%Y-%m-%d_%H%M%S}.jpg"
                snap.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with open(OUT / "events.csv", "a", newline="") as f:
                    csv.writer(f).writerow([ts.isoformat(timespec="seconds"), str(snap)])
                log("EVENT", snap)
                notify.push(f"Puppy squat {ts:%H:%M:%S}", snap)
        if now - last_prune > 3600:
            last_prune = now
            removed = prune(OUT, MAX_GB * 1e9)
            if removed:
                log("pruned", removed)


def probe():
    import cv2
    from ultralytics import YOLO
    det = YOLO("yolov8n.pt")
    gen = frames(RTSP_URL)
    frame = next(gen)
    gen.close()
    t = time.time()
    r = det(frame, imgsz=IMGSZ, classes=[DOG], conf=CONF, verbose=False)[0]
    log(f"frame {frame.shape[1]}x{frame.shape[0]}, {len(r.boxes)} dog(s), {1000*(time.time()-t):.0f} ms")
    cv2.imwrite("/tmp/probe.jpg", r.plot())
    log("saved /tmp/probe.jpg")


def selftest():
    f = np.zeros((100, 200, 3), np.uint8)
    assert pick_dog([[0, 0, 10, 10], [0, 0, 50, 50], [0, 0, 90, 90]], [DOG, DOG, 0], [0.9, 0.8, 0.9]) == ((0, 0, 50, 50), 0.8)
    assert pick_dog([[0, 0, 50, 50]], [DOG], [CONF - 0.1]) == (None, 0.0)
    assert crop(f, (190, 90, 200, 100)).shape == (11, 11, 3)
    s = Streak(3, 100)
    assert [s.update(x, 0) for x in (1, 1, 0, 1, 1, 1)] == [False] * 5 + [True]
    assert s.update(True, 1) is False and s.update(True, 1) is False and s.update(True, 1) is False
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for day in ("2026-01-01", "2026-01-02", "2026-01-03"):
            for sub in ("crops", "frames"):
                (root / sub / day).mkdir(parents=True)
                (root / sub / day / "a.jpg").write_bytes(b"x" * 1000)
        assert prune(root, 4500) == ["2026-01-01"]
        assert prune(root, 1) == ["2026-01-02"], "always keeps the newest day"
    print("selftest ok")


if __name__ == "__main__":
    {"probe": probe, "selftest": selftest}.get(sys.argv[1] if len(sys.argv) > 1 else "", run)()
