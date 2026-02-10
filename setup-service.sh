#!/bin/bash
# Setup FastTrack as a systemd service

echo "Setting up FastTrack service..."

# Copy service file to systemd
sudo cp fasttrack.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service (auto-start on boot)
sudo systemctl enable fasttrack

# Start the service
sudo systemctl start fasttrack

# Show status
sudo systemctl status fasttrack --no-pager

echo ""
echo "✓ Service installed and started!"
echo ""
echo "Useful commands:"
echo "  ./restart.sh          - Restart the server"
echo "  sudo systemctl stop fasttrack    - Stop the server"
echo "  sudo systemctl status fasttrack  - Check server status"
echo "  sudo journalctl -u fasttrack -f  - View live logs"
