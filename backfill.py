#!/usr/bin/env python3
"""Rebuild the detection log from recordings, for when the detector config changes.

On the Mac, with the share mounted and the same config as the Pi:
    set -a; . ./config.local; set +a
    uv run --with ultralytics --with opencv-python-headless python3 backfill.py run /Volumes/nas/puppycam/recordings/2026-11-03_*.mkv
Writes backfill.csv (same columns as detections/<day>.csv). Then on the Pi:
    scp backfill.csv homepi:/tmp/ && ssh homepi 'cd /opt/puppy-detector && set -a && . ./config.env && venv/bin/python backfill.py merge /tmp/backfill.csv'
merge unions the rows into OUT_DIR/detections/<day>.csv (existing rows are kept).
    backfill.py selftest
"""
import csv
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

import detector
from find_events import seg_start

FPS = float(os.environ.get("BACKFILL_FPS", "1"))


def frames_from_file(path, fps):
    w, h = 1920, 1080
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", str(path)]).decode().strip().split(",")
    w, h = int(out[0]), int(out[1])
    n = w * h * 3
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                          "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=n)
    i = 0
    while len(buf := p.stdout.read(n)) == n:
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        i += 1
    p.wait()


def run(segments, out):
    from ultralytics import YOLO
    det = YOLO(detector.MODEL)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        for seg in segments:
            seg = Path(seg)
            start, t0, rows = seg_start(seg), time.time(), 0
            for off, frame in frames_from_file(seg, FPS):
                r = det(frame, imgsz=detector.IMGSZ, classes=detector.CLASSES, conf=detector.CONF, verbose=False)[0]
                box, score = detector.pick_dog(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
                if box:
                    ts = start + timedelta(seconds=off)
                    w.writerow([ts.isoformat(timespec="milliseconds"), *box, round(score, 3)])
                    rows += 1
            f.flush()
            print(f"{seg.name}: {rows} dog frames, {time.time() - t0:.0f} s", flush=True)


def merge(src, root):
    """Union src rows into root/detections/<day>.csv, sorted, deduped by timestamp."""
    by_day = {}
    for r in csv.reader(open(src)):
        by_day.setdefault(r[0][:10], []).append(r)
    for day, rows in by_day.items():
        f = root / "detections" / f"{day}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        old = list(csv.reader(open(f))) if f.exists() else []
        merged = sorted({r[0]: r for r in old + rows}.values(), key=lambda r: r[0])
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            csv.writer(fh).writerows(merged)
        os.replace(tmp, f)
        print(f"{f.name}: {len(old)} + {len(rows)} -> {len(merged)} rows")


def selftest():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "detections").mkdir()
        (root / "detections" / "2026-11-03.csv").write_text("2026-11-03T10:00:00.000,1,1,2,2,0.5\n2026-11-03T12:00:00.000,1,1,2,2,0.5\n")
        src = root / "b.csv"
        src.write_text("2026-11-03T11:00:00.000,3,3,4,4,0.9\n2026-11-03T12:00:00.000,9,9,9,9,0.9\n2026-11-04T01:00:00.000,5,5,6,6,0.9\n")
        merge(src, root)
        rows = list(csv.reader(open(root / "detections" / "2026-11-03.csv")))
        assert [r[0][11:13] for r in rows] == ["10", "11", "12"] and rows[2][1] == "9", rows
        assert (root / "detections" / "2026-11-04.csv").read_text().startswith("2026-11-04T01")
    print("selftest ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run(sys.argv[2:], "backfill.csv")
    elif cmd == "merge":
        merge(sys.argv[2], detector.OUT)
    elif cmd == "selftest":
        selftest()
    else:
        sys.exit(__doc__)
