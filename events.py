"""Event store: one JSON object per line in OUT_DIR/events.jsonl.
Fields: id, time, type (poop|pee|null), confirmed, dismissed, source (manual|classifier), thumb, clip (paths relative to OUT_DIR).
"""
import json
import os
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/srv/nas/share/puppycam"))
FILE = OUT / "events.jsonl"


def load():
    if not FILE.exists():
        return []
    return [json.loads(l) for l in FILE.read_text().splitlines() if l.strip()]


def save(events):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text("".join(json.dumps(e) + "\n" for e in events))


def append(**ev):
    ev = {"type": None, "confirmed": False, "dismissed": False, "source": "manual", "thumb": None, "clip": None, **ev}
    FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FILE, "a") as f:
        f.write(json.dumps(ev) + "\n")
    return ev


def update(id, **changes):
    evs = load()
    for e in evs:
        if e["id"] == id:
            e.update(changes)
    save(evs)


def remove(id):
    save([e for e in load() if e["id"] != id])
