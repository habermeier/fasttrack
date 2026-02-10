#!/bin/bash
# Restart the FastTrack server

echo "Restarting FastTrack server..."
sudo systemctl restart fasttrack
sudo systemctl status fasttrack --no-pager
echo "Done!"
