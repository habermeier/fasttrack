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
python3 << 'PYEOF'
import json
import renderer

DATA_FILE = "telemetry.json"
CHART_FILE = "chart.png"

print("Reading data...")
with open(DATA_FILE, "r") as f:
    data = json.load(f)

print(f"Generating chart with {len(data)} data blocks...")
renderer.generate_chart(data, CHART_FILE)
print(f"✅ Chart saved to {CHART_FILE}")
PYEOF

# Show chart file info
echo ""
ls -lh chart.png
echo ""
echo "✅ Done! Chart regenerated."
echo ""
echo "View at: http://143.198.34.57/api/graph?t=$(date +%s)"
echo "(The ?t=$(date +%s) prevents browser caching)"
