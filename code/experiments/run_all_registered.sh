#!/bin/sh
# Run every registered configuration in the registered family order (WP5-v20 design v1 section 11; addendum A section 6).
# Usage: sh run_all_registered.sh <runs_root> [jobs]
ROOT=$1; JOBS=${2:-2}; HERE=$(cd "$(dirname "$0")" && pwd)
for FAM in EX1a EX2a EX4 EX1b EX5 EX6 EX7 EX8a EX8b EX3 EX2b EX2c EX2d EX1c EX1d; do
  echo "$(date '+%F %T') start $FAM"; python3 "$HERE/run_family.py" "$ROOT" --family $FAM --jobs $JOBS; echo "$(date '+%F %T') end $FAM"
done
for FAM in EX11 EX10 EX9; do
  echo "$(date '+%F %T') start $FAM"; python3 "$HERE/run_family_addA.py" "$ROOT" --family $FAM --jobs $JOBS; echo "$(date '+%F %T') end $FAM"
done
echo "$(date '+%F %T') ALL DONE"
