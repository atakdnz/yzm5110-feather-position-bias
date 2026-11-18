#!/bin/bash
# Fills the result tables from results/summary.json and renders the submission PDF with headless Chrome.
set -eu
cd "$(dirname "$0")/.."
python3 report/fill.py
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/Akdeniz_YZM5110_Research.pdf" "file://$PWD/report/report_filled.html" 2>/dev/null
ls -la Akdeniz_YZM5110_Research.pdf
