#!/usr/bin/env bash
# Deploy to the Pi and (re)start the service. Usage: ./sync.sh [host]
set -euo pipefail
cd "$(dirname "$0")"
host=${1:-homepi}
rsync -az detector.py notify.py train.py puppy-detector.service "$host:/opt/puppy-detector/"
[ -f config.local ] && rsync -az config.local "$host:/opt/puppy-detector/config.env"
ssh "$host" 'sudo install -m644 /opt/puppy-detector/puppy-detector.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now puppy-detector && sudo systemctl restart puppy-detector && sleep 3 && systemctl --no-pager status puppy-detector | head -12'
