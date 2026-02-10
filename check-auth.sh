#!/bin/bash
# Check cookie auth status

echo "🔍 Checking cookie authentication setup..."
echo ""

# Check if key directory exists
if [ -d "key" ]; then
    echo "✅ /key directory exists"
    ls -la key/
else
    echo "❌ /key directory does NOT exist"
fi

echo ""

# Check if latched file exists
if [ -f "key/latched" ]; then
    echo "✅ /key/latched file exists"
    echo "   Content length: $(wc -c < key/latched) characters"
    echo "   First 10 chars: $(head -c 10 key/latched)..."
else
    echo "❌ /key/latched file does NOT exist"
fi

echo ""
echo "🔍 Checking recent logs for auth errors..."
sudo journalctl -u fasttrack -n 100 --no-pager | grep -iE "error|exception|unauthorized|cookie|latch|key" | tail -10

echo ""
echo "🔍 Checking if middleware is loaded..."
sudo journalctl -u fasttrack -n 200 --no-pager | grep -iE "CookieRotation|startup" | tail -5

echo ""
echo "💡 To test: Visit the site and check again"
echo "   The key should be created on first visit"
