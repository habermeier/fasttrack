# Security Implementation: Cookie Rotation Auth

## Overview
This app uses a **cookie rotation system** to restrict write access to a single latched device (your phone) while keeping the site world-readable.

## How It Works

### First Visit (Latching)
1. When you first visit from your phone (no `/key/latched` file exists):
   - Server generates a secure 64-char random token
   - Saves it to `/key/latched`
   - Sets it as HTTP-only cookie
   - **Your device is now latched**

### Subsequent Visits (Cookie Rotation)
1. When you visit from your latched phone:
   - Server checks: cookie matches `/key/latched`?
   - If ✅ match:
     - Generates NEW random token
     - Updates `/key/latched` with new token
     - Sends new token as cookie
     - **Key rotates on every page load**

2. When someone else visits:
   - No cookie OR wrong cookie
   - Server doesn't send them a cookie
   - **Read-only access only**

## Protected Routes

### Entry Pages (Require Auth)
- `/` - Mobile Dashboard → redirects to `/graph` if unauthorized
- `/meal` - Log Meal → redirects to `/graph` if unauthorized

### Read-Only Pages
- `/graph` - High-Res Analytics (always accessible)

### API Endpoints
- **Write operations** (POST/PUT/DELETE to `/api/telemetry`): Require valid auth cookie
- **Read operations** (GET): World-readable

## Security Features

✅ **Rotating keys** - Old intercepted cookies become invalid immediately
✅ **HTTP-only cookies** - JavaScript can't access them
✅ **Device pinning** - Only first device gets latched
✅ **World-readable** - Anyone can view graphs/data
✅ **Single-writer** - Only you can add/modify data

## Resetting the Latch

If you need to latch a different device:
```bash
rm /home/bernie/fasttrack/key/latched
```

The next device to visit will become the new latched device.

## Cookie Details
- **Name**: `auth_token`
- **Type**: HTTP-only, SameSite=Strict
- **Max Age**: 1 year
- **Format**: 64-character hex token (cryptographically secure)

## Backward Compatibility
The old `sillykey` system is still present but no longer used for authentication. The `/api/config` endpoint still returns it for backward compatibility with existing client code.
