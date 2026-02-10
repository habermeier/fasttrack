# Deployment Guide

## Initial Setup

1. **Install the systemd service:**
   ```bash
   ./setup-service.sh
   ```

   This will:
   - Install FastTrack as a system service
   - Enable auto-start on boot
   - Start the server
   - Set up auto-restart on crash

## Common Operations

### Restart the server
```bash
./restart.sh
```

### Stop the server
```bash
sudo systemctl stop fasttrack
```

### Start the server
```bash
sudo systemctl start fasttrack
```

### Check server status
```bash
sudo systemctl status fasttrack
```

### View live logs
```bash
sudo journalctl -u fasttrack -f
```

### View recent logs
```bash
sudo journalctl -u fasttrack -n 100
```

## Auto-Update Behavior

The server automatically:
- ✅ **Pulls git changes every 60 seconds**
- ✅ **Hot-reloads `renderer.py`** when it changes
- ✅ **Auto-restarts when `main.py` changes**
- ✅ **Regenerates charts** when data or renderer changes
- ✅ **Restarts on crash** (via systemd)

## Manual Deployment

If you need to deploy changes manually:

1. **Push changes to git:**
   ```bash
   git push origin master
   ```

2. **Server auto-pulls within 60 seconds**
   - If only `renderer.py` changed → hot-reload
   - If `main.py` changed → auto-restart
   - If `telemetry.json` changed → regenerate chart

3. **Or restart immediately:**
   ```bash
   ./restart.sh
   ```

## Troubleshooting

### Server won't start
```bash
# Check logs
sudo journalctl -u fasttrack -n 50

# Check if port 80 is in use
sudo lsof -i :80

# Check file permissions
ls -la /home/fasttrack/app/
```

### Service not found
```bash
# Reinstall service
./setup-service.sh
```

### Permission issues
The service runs as user `fasttrack`. Ensure:
- User exists: `id fasttrack`
- User can sudo systemctl: edit `/etc/sudoers.d/fasttrack`
- Files are owned by user: `sudo chown -R fasttrack:fasttrack /home/fasttrack/app/`
