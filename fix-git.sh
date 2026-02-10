#!/bin/bash
# Fix git issues and force sync with remote

echo "🔧 Fixing git state..."
echo ""

# Show current status
echo "Current git status:"
git status
echo ""

# Stash any local changes
echo "Stashing local changes..."
git stash
echo ""

# Reset to remote state
echo "Resetting to remote master..."
git fetch origin
git reset --hard origin/master
echo ""

# Clean untracked files (but preserve data files)
echo "Cleaning up..."
git clean -fd -e telemetry.json -e chart.png -e server.log -e key/
echo ""

# Show final status
echo "✅ Git state fixed!"
git log --oneline -3
echo ""
echo "Now restart the server:"
echo "  sudo systemctl restart fasttrack"
