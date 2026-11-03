#!/usr/bin/env python3
"""Find the moment of an accident after the fact.

    find_events.py find --since "3 hours ago" [--until "1 hour ago"] [--min-seconds 4] [--radius 80] [--top 15]
        scans puppycam/detections/*.csv for runs where the dog stayed in one spot, ranks them by
        duration and writes a contact sheet per candidate to puppycam/candidates/<id>_<secs>s.jpg
    find_events.py clip <id> [--pad 10]
        cuts that run (+pad seconds each side) from the recordings into puppycam/positives/<id>.mp4
        and saves dog crops from it to puppycam/positives/<id>/ for training
    find_events.py selftest

Times: "N minutes ago", "N hours ago", "HH:MM" (today), or ISO "2026-11-03 14:30".
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/srv/nas/share/puppycam"))
REC = OUT / "recordings"
SEG_SECONDS = 600
GAP = 2.5  # seconds without a detection that ends a run (detector runs at 2 fps)


def parse_time(s, now=None):
    now = now or datetime.now()
    m = re.fullmatch(r"(\d+)\s*(min\w*|m|h\w*)\s*ago", s.strip(), re.I)
    if m:
        n, unit = int(m[1]), m[2].lower()
        return now - (timedelta(hours=n) if unit.startswith("h") else timedelta(minutes=n))
    if re.fullmatch(r"\d{1,2}:\d{2}", s.strip()):
        h, mi = map(int, s.split(":"))
        return now.replace(hour=h, minute=mi, second=0, microsecond=0)
    return datetime.fromisoformat(s.strip())


def load_detections(root, start, end):
    rows = []
    d = start.date()
    while d <= end.date():
        f = root / "detections" / f"{d}.csv"
        if f.exists():
            for r in csv.reader(open(f)):
                t = datetime.fromisoformat(r[0])
                if start <= t <= end:
                    rows.append((t, *(int(v) for v in r[1:5]), float(r[5])))
        d += timedelta(days=1)
    return rows


def find_runs(rows, radius, min_seconds):
    """Group consecutive detections whose bbox centre stays within radius px of the run's mean centre."""
    runs, cur = [], []

    def flush():
        if len(cur) >= 3 and (cur[-1][0] - cur[0][0]).total_seconds() >= min_seconds:
            runs.append(list(cur))
        cur.clear()

    cx = cy = 0.0
    for r in rows:
        x, y = (r[1] + r[3]) / 2, (r[2] + r[4]) / 2
        if cur and ((r[0] - cur[-1][0]).total_seconds() > GAP or math.hypot(x - cx, y - cy) > radius):
            flush()
        cur.append(r)
        n = len(cur)
        cx, cy = cx + (x - cx) / n, cy + (y - cy) / n
    flush()
    runs.sort(key=lambda run: (run[-1][0] - run[0][0]), reverse=True)
    return runs


def run_id(run):
    return run[0][0].strftime("%Y-%m-%d_%H%M%S")


def seg_start(p):
    return datetime.strptime(p.stem, "%Y-%m-%d_%H%M%S")


def segment_for(t):
    segs = sorted(REC.glob("20*.mkv"))
    before = [s for s in segs if seg_start(s) <= t]
    if before and (t - seg_start(before[-1])).total_seconds() < SEG_SECONDS + 60:
        return before[-1]
    return None


def frame_at(t, out_jpg, width=480):
    seg = segment_for(t)
    if seg is None:
        return False
    off = (t - seg_start(seg)).total_seconds()
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{off:.2f}", "-i", str(seg),
                        "-frames:v", "1", "-vf", f"scale={width}:-2", "-y", str(out_jpg)], capture_output=True)
    return r.returncode == 0 and Path(out_jpg).exists()


def contact_sheet(run, dst, n=6):
    import cv2
    import numpy as np
    t0, t1 = run[0][0], run[-1][0]
    tiles = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            t = t0 + (t1 - t0) * i / max(1, n - 1)
            p = Path(td) / f"{i}.jpg"
            if frame_at(t, p):
                img = cv2.imread(str(p))
                cv2.putText(img, t.strftime("%H:%M:%S"), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                tiles.append(img)
    if not tiles:
        return False
    h, w = tiles[0].shape[:2]
    tiles = [cv2.resize(t, (w, h)) for t in tiles]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return True


def find(since, until=None, radius=80, min_seconds=4, top=15):
    """Candidates in the window as dicts: id, start, secs, centre, sheet (path relative to OUT or None)."""
    now = datetime.now()
    start, end = parse_time(since, now), parse_time(until, now) if until else now
    out = []
    for run in find_runs(load_detections(OUT, start, end), radius, min_seconds)[:top]:
        secs = (run[-1][0] - run[0][0]).total_seconds()
        cx = sum((r[1] + r[3]) / 2 for r in run) / len(run)
        cy = sum((r[2] + r[4]) / 2 for r in run) / len(run)
        sheet = OUT / "candidates" / f"{run_id(run)}_{secs:.0f}s.jpg"
        ok = sheet.exists() or contact_sheet(run, sheet)
        out.append({"id": run_id(run), "start": run[0][0].isoformat(timespec="seconds"), "secs": round(secs),
                    "centre": [round(cx), round(cy)], "sheet": str(sheet.relative_to(OUT)) if ok else None})
    return out


def clip(id, pad=10, radius=80):
    """Cut the run starting at id (+pad s each side) into positives/<id>.mp4, save its dog crops. Returns (clip, n_crops)."""
    t0 = datetime.strptime(id, "%Y-%m-%d_%H%M%S")
    rows = load_detections(OUT, t0 - timedelta(seconds=1), t0 + timedelta(hours=1))
    run = next((r for r in find_runs(rows, radius, 0) if run_id(r) == id), None)
    if run is None:
        raise ValueError(f"no run starting at {id}; ids come from `find`")
    start, end = run[0][0] - timedelta(seconds=pad), run[-1][0] + timedelta(seconds=pad)
    segs = [s for s in sorted(REC.glob("20*.mkv")) if seg_start(s) <= end and seg_start(s) + timedelta(seconds=SEG_SECONDS) >= start]
    if not segs:
        raise ValueError("no recording covers that window")
    dst = OUT / "positives" / f"{id}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
        lst.writelines(f"file '{s}'\n" for s in segs)
    off = (start - seg_start(segs[0])).total_seconds()
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst.name,
                    "-ss", f"{max(0, off):.2f}", "-t", f"{(end - start).total_seconds():.2f}", "-c", "copy", "-y", str(dst)], check=True)
    os.unlink(lst.name)
    return dst, save_crops(dst, OUT / "positives" / id)


def cmd_find(a):
    cands = find(a.since, a.until, a.radius, a.min_seconds, a.top)
    print(f"{len(cands)} candidates\n{'id':20} {'secs':>5} {'centre':>12}  sheet")
    for c in cands:
        print(f"{c['id']:20} {c['secs']:5} {c['centre'][0]:6},{c['centre'][1]:5}  {c['sheet'] or '(no recording for this time)'}")
    if cands:
        print(f"\nsheets in {OUT / 'candidates'}. Then: find_events.py clip <id>")


def cmd_clip(a):
    try:
        dst, n = clip(a.id, a.pad, a.radius)
    except ValueError as e:
        sys.exit(str(e))
    print(f"clip: {dst}\n{n} crops in {dst.with_suffix('')}  (train.py picks these up)")


def save_crops(video, dst, every=7):
    import cv2
    from ultralytics import YOLO
    import detector
    det = YOLO(str(Path(__file__).with_name("yolov8n.pt")))
    cap, i, n = cv2.VideoCapture(str(video)), 0, 0
    dst.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % every:
            continue
        r = det(frame, imgsz=detector.IMGSZ, classes=[detector.DOG], conf=detector.CONF, verbose=False)[0]
        box, _ = detector.pick_dog(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
        if box:
            n += 1
            cv2.imwrite(str(dst / f"{n:04d}.jpg"), detector.crop(frame, box), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return n


def selftest():
    now = datetime(2026, 11, 3, 12, 0, 0)
    assert parse_time("3 hours ago", now) == now - timedelta(hours=3)
    assert parse_time("45 min ago", now) == now - timedelta(minutes=45)
    assert parse_time("09:30", now) == now.replace(hour=9, minute=30)
    assert parse_time("2026-11-03 08:00", now) == datetime(2026, 11, 3, 8)
    t = lambda s: now + timedelta(seconds=s)  # noqa: E731
    still = [(t(s * 0.5), 100, 100, 200, 200, 0.9) for s in range(12)]           # 5.5 s in one spot
    walk = [(t(20 + s * 0.5), 100 + s * 60, 100, 200 + s * 60, 200, 0.9) for s in range(12)]  # moving
    short = [(t(60 + s * 0.5), 500, 500, 600, 600, 0.9) for s in range(4)]       # 1.5 s only
    gap = [(t(100 + s * 0.5), 300, 300, 400, 400, 0.9) for s in range(6)] + [(t(110 + s * 0.5), 300, 300, 400, 400, 0.9) for s in range(6)]
    runs = find_runs(still + walk + short + gap, radius=80, min_seconds=4)
    assert [run_id(r) for r in runs] == ["2026-11-03_120000"], [run_id(r) for r in runs]
    runs = find_runs(gap, radius=80, min_seconds=2)
    assert len(runs) == 2, "a 4 s gap splits the run"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "detections").mkdir()
        with open(root / "detections" / "2026-11-03.csv", "w", newline="") as f:
            csv.writer(f).writerows([[r[0].isoformat(timespec="milliseconds"), *r[1:]] for r in still])
        assert len(load_detections(root, t(1), t(3))) == 5
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find")
    f.add_argument("--since", required=True)
    f.add_argument("--until")
    f.add_argument("--min-seconds", type=float, default=4)
    f.add_argument("--radius", type=float, default=80)
    f.add_argument("--top", type=int, default=15)
    c = sub.add_parser("clip")
    c.add_argument("id")
    c.add_argument("--pad", type=float, default=10)
    c.add_argument("--radius", type=float, default=80)
    sub.add_parser("selftest")
    a = p.parse_args()
    {"find": cmd_find, "clip": cmd_clip, "selftest": lambda a: selftest()}[a.cmd](a)
