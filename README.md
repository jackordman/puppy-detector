# puppy-detector

Watches the Wyze Cam v3 from the Raspberry Pi `homepi`, saves a crop whenever a dog is in
frame, and (once trained) pushes a notification when the dog squats to poop or lifts a leg.
No sound, no siren, ever. Output is a push to Jack's phone plus a log row and snapshot.

Everything lands on the NAS share at `smb://homepi.local/nas` under `puppycam/`:

    puppycam/crops/YYYY-MM-DD/HHMMSS_mmm.jpg    dog crop, max one per second
    puppycam/frames/YYYY-MM-DD/HHMMSS_mmm.jpg   matching full frame
    puppycam/detections/YYYY-MM-DD.csv           one row per frame with a dog: time, x1,y1,x2,y2, conf
    puppycam/recordings/YYYY-MM-DD_HHMMSS.mkv    ring buffer, 10 min segments, ~15 GB/day, capped at 250 GB
    puppycam/candidates/<id>_<secs>s.jpg         contact sheets from `puppy find`
    puppycam/positives/<id>.mp4 + <id>/*.jpg     clips and their dog crops from `puppy clip` (training input)
    puppycam/label/squat/                        phase 2: positives you sorted
    puppycam/label/other/                        phase 2: optional hand-picked negatives
    puppycam/events/                             phase 3: snapshot per event
    puppycam/events.csv                          phase 3: time, snapshot path

`crops/` and `frames/` are pruned oldest-day-first above `MAX_GB` (20 GB), `recordings/`
oldest-segment-first above `MAX_REC_GB` (250 GB, about two weeks). Everything else is kept.

## Layout

- `detector.py` the service. ffmpeg pulls the RTSPS stream at 2 fps, YOLOv8n (CPU, 416 px)
  finds the dog, OpenCV writes the JPEGs. Loads `squat.pt` if present and fires an event after
  3 consecutive positive frames, 2 min cooldown.
- `puppy-recorder.service` ffmpeg copying the h264 stream into 10 minute `.mkv` segments, no
  re-encode, no audio. Its own unit so it and the detector cannot take each other down.
- `puppy-prune.timer` runs `prune.sh` every 10 minutes to hold the recordings under the cap.
- `find_events.py` and the `puppy` wrapper: find the accident after the fact, cut the clip.
- `train.py` phase 2 trainer (YOLOv8n-cls fine-tune on the sorted crops and clipped positives).
- `notify.py` ntfy.sh push with the snapshot attached.
- `puppy-detector.service`, `sync.sh` deploy. `config.example` documents every knob.

Pi side: `/opt/puppy-detector/` holds the code, a venv with `ultralytics` and
`opencv-python-headless`, `config.env` (the RTSP URL with credentials, not in git) and the
downloaded weights.

## Deploy

    cp config.example config.local   # set RTSP_URL (same one as wyze-cam-zoom/config.local)
    ./sync.sh                        # rsync + install service + restart

    ssh homepi journalctl -fu puppy-detector
    ssh homepi systemctl status puppy-detector puppy-recorder puppy-prune.timer
    ssh homepi 'cd /opt/puppy-detector && venv/bin/python detector.py probe'   # one frame, timing

Self-checks: `python3 detector.py selftest` (needs numpy, e.g. `uv run --with numpy ...`),
`python3 train.py selftest`, `python3 find_events.py selftest`.

The Pi needs the CPU-only torch wheel. Plain `pip install ultralytics` pulls the aarch64 CUDA
build, which SIGILLs on the Pi 4's Cortex-A72. After creating the venv:

    venv/bin/pip install ultralytics opencv-python-headless
    venv/bin/pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch torchvision

## Phase 1, collect (now)

Let it run one to two weeks. Browse `puppycam/crops/` from the Mac to confirm the dog is
being caught and the crops look usable.

## Finding an accident after the fact

You find the mess an hour later and want the moment. From the Mac:

    ./puppy find --since "3 hours ago"               # or --since 13:00 --until 14:30
    ./puppy clip 2026-11-03_143012                   # id from the table, --pad 10 default

`find` scans the detection log for runs where the dog stayed within `--radius` 80 px for at
least `--min-seconds` 4, ranks them longest first and writes a six-frame contact sheet per
candidate to `puppycam/candidates/`. Look at the sheets in Finder, pick the right id. `clip`
cuts that window out of the recordings into `puppycam/positives/<id>.mp4` and saves the dog
crops from it to `puppycam/positives/<id>/`. Those crops are the training positives for phase 2.

## Phase 2, label and train

1. Collect positives with `puppy clip` (above). For more, or for events you saw live, move
   crops from `puppycam/crops/<day>/` into `puppycam/label/squat/`. Aim for 50+ positives.
   Negatives are sampled from the remaining crops automatically; drop hard negatives
   (sitting, sniffing) into `puppycam/label/other/` if the classifier confuses them.
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
