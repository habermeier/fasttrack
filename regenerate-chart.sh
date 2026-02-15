#!/bin/bash
# Manually regenerate the chart

echo "🎨 Regenerating chart..."
echo ""

# Check if we're in the right directory
if [ ! -f "renderer.py" ]; then
    echo "❌ Error: Not in app directory"
    exit 1
fi

# Run Python to regenerate
venv/bin/python3 << 'PYEOF'
import json
import renderer

DATA_FILE = "telemetry.json"
CHART_FILE = "chart.png"
CHART_FILE_SVG = "chart.svg"

print("Reading data...")
with open(DATA_FILE, "r") as f:
    data = json.load(f)

print(f"Generating charts with {len(data)} data blocks...")
renderer.generate_chart(data, CHART_FILE)
renderer.generate_chart(data, CHART_FILE_SVG)
print(f"✅ Charts saved to {CHART_FILE} and {CHART_FILE_SVG}")
PYEOF

# Show chart file info
echo ""
ls -lh chart.png chart.svg
echo ""
echo "✅ Done! Charts regenerated."
echo ""
echo "View at: http://143.198.34.57/api/graph.svg?t=$(date +%s)"
echo "(The ?t=$(date +%s) prevents browser caching)"
