#!/bin/bash
# Check recent logs for auto-pull worker evidence

echo "🔍 Checking recent logs for auto-pull worker evidence..."
echo ""

# Check last 200 log lines for auto-pull activity
FOUND=$(sudo journalctl -u fasttrack -n 200 --no-pager | grep -E "Auto-pull|Remote changes|fetch.*origin|Synchronizing" | tail -1)

if [ -n "$FOUND" ]; then
    echo "✅ FOUND IT!"
    echo ""
    echo "Most recent auto-pull activity:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$FOUND"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📍 Location: systemd journal for fasttrack.service"
    echo "   View full logs: sudo journalctl -u fasttrack -f"
    echo ""

    # Extract and show timestamp
    TIMESTAMP=$(echo "$FOUND" | awk '{print $1, $2, $3}')
    echo "⏰ Last seen: $TIMESTAMP"
else
    echo "❌ No auto-pull activity found in recent logs."
    echo ""
    echo "Checking if service is running..."
    if sudo systemctl is-active --quiet fasttrack; then
        echo "✅ Service is running"
        echo ""
        echo "Possible reasons:"
        echo "  - Service just started (auto-pull runs every 60s)"
        echo "  - No remote changes to pull yet"
        echo "  - Worker thread didn't start (check startup logs)"
        echo ""
        echo "Run this to wait for next auto-pull cycle:"
        echo "  ./check-autopull.sh"
    else
        echo "❌ Service is NOT running!"
        echo ""
        echo "Start it with: sudo systemctl start fasttrack"
        echo "Or run setup: ./setup-service.sh"
    fi
fi
