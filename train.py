#!/usr/bin/env python3
"""Phase 2: train the squat classifier from crops sorted by hand on the share.

Positives: OUT_DIR/positives/*/*.jpg (from `puppy clip`) plus OUT_DIR/label/squat/*.jpg (sorted by hand).
Negatives: OUT_DIR/label/other/*.jpg if any, else a random sample of OUT_DIR/crops/.
Writes best weights to CLASSIFIER (default /opt/puppy-detector/squat.pt).

    train.py            train (run on the Pi: ssh homepi 'cd /opt/puppy-detector && venv/bin/python train.py')
    train.py selftest   checks the dataset builder with fake files
"""
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/srv/nas/share/puppycam"))
CLASSIFIER = Path(os.environ.get("CLASSIFIER", "/opt/puppy-detector/squat.pt"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))


def build_dataset(root, dst, neg_ratio=3, val_frac=0.2, seed=0):
    """Lay out dst/{train,val}/{squat,other}/ as symlinks. Returns (n_pos, n_neg)."""
    pos = sorted((root / "label" / "squat").glob("*.jpg")) + sorted(root.glob("positives/*/*.jpg"))
    neg = sorted((root / "label" / "other").glob("*.jpg"))
    if not neg:
        pool = sorted(root.glob("crops/*/*.jpg"))
        random.Random(seed).shuffle(pool)
        neg = pool[: len(pos) * neg_ratio]
    if len(pos) < 20 or not neg:
        raise SystemExit(f"need >=20 positives and some negatives, have {len(pos)} / {len(neg)}")
    for name, files in (("squat", pos), ("other", neg)):
        files = list(files)
        random.Random(seed).shuffle(files)
        k = max(1, int(len(files) * val_frac))
        for split, part in (("val", files[:k]), ("train", files[k:])):
            d = dst / split / name
            d.mkdir(parents=True, exist_ok=True)
            for f in part:
                (d / f"{f.parent.name}_{f.name}").symlink_to(f.resolve())
    return len(pos), len(neg)


def train():
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory() as td:
        n = build_dataset(OUT, Path(td))
        print("dataset squat/other:", n)
        model = YOLO("yolov8n-cls.pt")
        r = model.train(data=td, epochs=EPOCHS, imgsz=224, batch=32, workers=2, project="/tmp/squat-train", exist_ok=True)
        best = Path(r.save_dir) / "weights" / "best.pt"
        shutil.copy(best, CLASSIFIER)
        print("wrote", CLASSIFIER, "- restart the service: sudo systemctl restart puppy-detector")


def selftest():
    with tempfile.TemporaryDirectory() as td:
        root, dst = Path(td) / "share", Path(td) / "ds"
        (root / "label" / "squat").mkdir(parents=True)
        (root / "crops" / "2026-01-01").mkdir(parents=True)
        (root / "positives" / "2026-01-01_120000").mkdir(parents=True)
        for i in range(20):
            (root / "label" / "squat" / f"p{i}.jpg").write_bytes(b"x")
        for i in range(5):
            (root / "positives" / "2026-01-01_120000" / f"{i:04d}.jpg").write_bytes(b"x")
        for i in range(200):
            (root / "crops" / "2026-01-01" / f"n{i}.jpg").write_bytes(b"x")
        assert build_dataset(root, dst) == (25, 75)
        assert len(list((dst / "val" / "squat").iterdir())) == 5
        assert len(list((dst / "train" / "other").iterdir())) == 60
        assert next((dst / "train" / "squat").iterdir()).resolve().exists()
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else train()
