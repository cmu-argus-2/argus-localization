#!/bin/bash
# Copy this package's scripts to the workstation or the Jetson, from your Mac.
#   ./sync.sh workstation   -> ~/argus-localization/deploy/jetson/ on the workstation
#   ./sync.sh jetson        -> ~/Desktop/astroloc/ on the Jetson (via the workstation)
# Files are streamed over ssh (the Jetson has no internet and is reached through the workstation).
# Lists what it will copy and asks before writing.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
FILES=$(ls *.py *.sh | grep -v -e '^sync.sh$' -e '^config.sh$')
FILES="$FILES $(ls two_frame/*.py od/*.py)"
case "${1:-}" in
  workstation) DEST_DESC="$WS:~/$WS_REPO/deploy/jetson/"; ;;
  jetson)      DEST_DESC="$JETSON:~/$JETSON_DIR/ (via $WS)"; ;;
  *) echo "usage: ./sync.sh workstation|jetson"; exit 1 ;;
esac
echo "Will copy to $DEST_DESC:"; echo "$FILES" | tr ' ' '\n' | sed 's/^/  /'
read -r -p "Proceed? [y/N] " ok; [ "$ok" = "y" ] || exit 0
if [ "$1" = workstation ]; then
  tar -cf - $FILES | ssh "$WS" "mkdir -p ~/$WS_REPO/deploy/jetson && tar -C ~/$WS_REPO/deploy/jetson -xf -"
else
  tar -cf - $FILES | ssh -J "$WS" "$JETSON" "mkdir -p ~/$JETSON_DIR && tar -C ~/$JETSON_DIR -xf -"
fi
echo "done"
