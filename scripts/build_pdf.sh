#!/bin/bash
# Renders report/report.html to the submission PDF with headless Google Chrome.
set -eu
cd "$(dirname "$0")/.."
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/Akdeniz_YZM5110_Research.pdf" "file://$PWD/report/report.html" 2>/dev/null
ls -la Akdeniz_YZM5110_Research.pdf
