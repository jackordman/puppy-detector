#!/usr/bin/env bash
# Deploy to the Pi and (re)start the services. Usage: ./sync.sh [host]
# Restarting the recorder drops a few seconds of recording at the segment boundary.
set -euo pipefail
cd "$(dirname "$0")"
host=${1:-homepi}
rsync -az *.py *.sh puppy-*.service puppy-*.timer "$host:/opt/puppy-detector/"
[ -f config.local ] && rsync -az config.local "$host:/opt/puppy-detector/config.env"
ssh "$host" 'cd /opt/puppy-detector && sudo install -m644 puppy-*.service puppy-*.timer /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now puppy-detector puppy-recorder puppy-web puppy-prune.timer && sudo systemctl restart puppy-detector puppy-recorder puppy-web && sleep 4 && systemctl --no-pager --lines=5 status puppy-detector puppy-recorder puppy-web puppy-prune.timer'
