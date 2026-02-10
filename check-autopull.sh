#!/bin/bash
# Check if auto-pull worker is running and active

echo "🔍 Checking for auto-pull worker activity..."
echo "   Watching logs for up to 90 seconds..."
echo ""

# Watch logs for auto-pull activity with timeout
timeout 90s sudo journalctl -u fasttrack -f -n 0 | while read -r line; do
    # Check for any auto-pull related activity
    if echo "$line" | grep -qE "Auto-pull|Remote changes|fetch.*origin|Synchronizing"; then
        echo "✅ FOUND IT!"
        echo ""
        echo "Auto-pull worker is active. Found this log entry:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "$line"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "📍 Location: systemd journal for fasttrack.service"
        echo "   View full logs: sudo journalctl -u fasttrack -f"
        exit 0
    fi
done

# If we get here, timeout occurred
echo "⏱️  No auto-pull activity detected in 90 seconds."
echo ""
echo "This could mean:"
echo "  1. The service isn't running (check: sudo systemctl status fasttrack)"
echo "  2. No remote changes in the last 90 seconds (normal)"
echo "  3. Auto-pull worker crashed (check: sudo journalctl -u fasttrack -n 50)"
echo ""
echo "To see recent logs:"
echo "  sudo journalctl -u fasttrack -n 100"
