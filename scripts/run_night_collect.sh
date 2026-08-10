#!/bin/bash
# CATI night collection runner — called by cron, no manual intervention needed.
# Usage: run_night_collect.sh <stop_hour>
#   stop_hour: decimal SGT hour to stop (AM = +24). 24.5 = 00:30, 30.5 = 06:30
#
# Cron schedule (IST):
#   0 16 * * *  →  18:30 SGT evening job  →  stop_hour=24.5
#   0 22 * * *  →  00:30 SGT dawn job     →  stop_hour=30.5
#
# No Drive auth needed — images are saved to Colab /content/ then pushed to HF.
# Requires ~/.cache/huggingface/token to exist (run `huggingface-cli login` once).

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

# Create CPU session
$COLAB new --session "$SESSION"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create Colab session"
    exit 1
fi

# Upload HF token as a file to avoid shell escaping issues with the token string
HF_TOK_FILE=$(mktemp /tmp/hf_tok.XXXXXX)
cat ~/.cache/huggingface/token > "$HF_TOK_FILE"
$COLAB upload -s "$SESSION" "$HF_TOK_FILE" /content/.hf_token
rm "$HF_TOK_FILE"

# Set env vars in the kernel — persists for the session lifetime
printf "import os\nos.environ['CATI_STOP_HOUR'] = '%s'\nos.environ['HF_TOKEN'] = open('/content/.hf_token').read().strip()\nos.environ['HF_RAW_REPO'] = 'SuhxsReddy/cati-night-raw'\n" "$STOP_HOUR" | \
    $COLAB exec -s "$SESSION"

# Run collection — caffeinate keeps Mac awake, timeout matches full window + upload buffer
caffeinate -i $COLAB exec -s "$SESSION" \
    --file "$SCRIPT_LOCAL" \
    --timeout 23400

echo ""
echo "Finished  : $(date)"

$COLAB stop -s "$SESSION" || true
