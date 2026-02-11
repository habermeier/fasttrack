#!/bin/bash
# Manually sync telemetry.json to git

echo "📤 Syncing telemetry.json..."
echo ""

cd /home/fasttrack/app

# Check git status
echo "Current git status:"
git status telemetry.json
echo ""

# Add and commit
git add telemetry.json
if git diff --cached --quiet telemetry.json; then
    echo "✓ No changes to commit"
else
    git commit -m "telemetry: manual sync $(date +%Y-%m-%d_%H:%M)"
    echo "✓ Committed"

    # Push
    git push origin master
    echo "✓ Pushed to remote"
fi

echo ""
echo "Done!"
