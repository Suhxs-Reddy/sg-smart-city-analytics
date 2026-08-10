#!/bin/bash
# CATI night collection runner — called by cron, no manual intervention needed.
# Usage: run_night_collect.sh <stop_hour>
#   stop_hour: decimal SGT hour to stop (AM = +24). 24.5 = 00:30, 30.5 = 06:30
#
# Cron schedule (IST):
#   0 16 * * *  →  18:30 SGT evening job  →  stop_hour=24.5
#   0 22 * * *  →  00:30 SGT dawn job     →  stop_hour=30.5

STOP_HOUR=${1:-31.5}
COLAB="/opt/anaconda3/bin/colab"
SCRIPT_LOCAL="/Users/suhasreddy/Documents/Computer vision Projects/sg-smart-city-analytics/scripts/collect_night_standalone.py"
LOG_DIR="/Users/suhasreddy/Documents/Computer vision Projects/sg-smart-city-analytics/logs"
SESSION="night-$(date +%Y%m%d-%H%M)"

mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/night-$(date +%Y%m%d-%H%M).log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "=== CATI Night Collect ==="
echo "Session   : $SESSION"
echo "Stop hour : $STOP_HOUR SGT"
echo "Started   : $(date)"
echo ""

# Create CPU session (no GPU needed for collection)
$COLAB new --session "$SESSION"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create Colab session"
    exit 1
fi

# Mount Google Drive on the VM
$COLAB drivemount -s "$SESSION"

# Upload the collection script
$COLAB upload -s "$SESSION" "$SCRIPT_LOCAL"

# Set stop hour in the VM kernel's environment
$COLAB exec -s "$SESSION" "import os; os.environ['CATI_STOP_HOUR']='$STOP_HOUR'"

# Run collection — caffeinate keeps Mac awake, timeout matches 6h window
caffeinate -i $COLAB exec -s "$SESSION" \
    --file "$SCRIPT_LOCAL" \
    --timeout 21300

echo ""
echo "Finished  : $(date)"

# Clean up session
$COLAB stop -s "$SESSION" || true
