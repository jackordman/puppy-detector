# puppy-detector

Watches the Wyze Cam v3 from the Raspberry Pi `homepi`, saves a crop whenever a dog is in
frame, and (once trained) pushes a notification when the dog squats to poop or lifts a leg.
No sound, no siren, ever. Output is a push to Jack's phone plus a log row and snapshot.

Everything lands on the NAS share at `smb://homepi.local/nas` under `puppycam/`:

    puppycam/crops/YYYY-MM-DD/HHMMSS_mmm.jpg    dog crop, max one per second
    puppycam/frames/YYYY-MM-DD/HHMMSS_mmm.jpg   matching full frame
    puppycam/label/squat/                        phase 2: positives you sorted
    puppycam/label/other/                        phase 2: optional hand-picked negatives
    puppycam/events/                             phase 3: snapshot per event
    puppycam/events.csv                          phase 3: time, snapshot path

`crops/` and `frames/` are pruned oldest-day-first above `MAX_GB` (20 GB). `label/` and
`events/` are never pruned.

## Layout

- `detector.py` the service. ffmpeg pulls the RTSPS stream at 2 fps, YOLOv8n (CPU, 416 px)
  finds the dog, OpenCV writes the JPEGs. Loads `squat.pt` if present and fires an event after
  3 consecutive positive frames, 2 min cooldown.
- `train.py` phase 2 trainer (YOLOv8n-cls fine-tune on the sorted crops).
- `notify.py` ntfy.sh push with the snapshot attached.
- `puppy-detector.service`, `sync.sh` deploy. `config.example` documents every knob.

Pi side: `/opt/puppy-detector/` holds the code, a venv with `ultralytics` and
`opencv-python-headless`, `config.env` (the RTSP URL with credentials, not in git) and the
downloaded weights.

## Deploy

    cp config.example config.local   # set RTSP_URL (same one as wyze-cam-zoom/config.local)
    ./sync.sh                        # rsync + install service + restart

    ssh homepi journalctl -fu puppy-detector
    ssh homepi 'cd /opt/puppy-detector && venv/bin/python detector.py probe'   # one frame, timing

Self-checks: `python3 detector.py selftest`, `python3 train.py selftest` (need numpy locally:
`uv run --with numpy python3 detector.py selftest`).

## Phase 1, collect (now)

Let it run one to two weeks. Browse `puppycam/crops/` from the Mac to confirm the dog is
being caught and the crops look usable.

## Phase 2, label and train

1. In Finder, go through `puppycam/crops/<day>/` and move every squat or leg-lift crop into
   `puppycam/label/squat/`. Aim for at least 50, more is better. Negatives are sampled from
   the remaining crops automatically; drop hard negatives (sitting, sniffing) into
   `puppycam/label/other/` if the classifier confuses them.
2. Train on the Pi (ultralytics is already there, 20 epochs at 224 px takes a while on a Pi 4,
   run it in tmux):

       ssh homepi 'cd /opt/puppy-detector && venv/bin/python train.py'

   It writes `/opt/puppy-detector/squat.pt`. Training on the Mac works too: install
   ultralytics in a venv, mount the share, run with `OUT_DIR=/Volumes/nas/puppycam
   CLASSIFIER=./squat.pt`, then `rsync squat.pt homepi:/opt/puppy-detector/`.
3. `ssh homepi sudo systemctl restart puppy-detector`. The journal shows
   `classifier: /opt/puppy-detector/squat.pt`.

Tune `SQUAT_CONF` and `CONSECUTIVE` in `config.local` against the false-positive rate, resync.

## Phase 3, notify

Pick a hard-to-guess topic name, set `NTFY_TOPIC=` in `config.local`, `./sync.sh`, subscribe to
the same topic in the ntfy app on the phone. Test with:

    ssh homepi 'cd /opt/puppy-detector && set -a && . ./config.env && venv/bin/python notify.py /tmp/probe.jpg'

Every event appends to `puppycam/events.csv` and saves the frame to `puppycam/events/`.
Pushover instead: swap the `push()` body in `notify.py` for a POST to
`api.pushover.net/1/messages.json` with the image as `attachment`.
