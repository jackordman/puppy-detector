#!/usr/bin/env bash
# Delete the oldest recording segments until the folder is under MAX_REC_GB. Never touches the newest.
set -euo pipefail
dir=${OUT_DIR:-/srv/nas/share/puppycam}/recordings
max_kb=$(( ${MAX_REC_GB:-250} * 1000000 ))
[ -d "$dir" ] || exit 0
while [ "$(du -sk "$dir" | cut -f1)" -gt "$max_kb" ]; do
  mapfile -t files < <(ls "$dir"/*.mkv 2>/dev/null | sort)
  [ "${#files[@]}" -gt 1 ] || break
  rm -f -- "${files[0]}"
  echo "pruned ${files[0]}"
done
