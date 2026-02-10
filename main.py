from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import json
import os
import hashlib
import secrets
from datetime import datetime
import uuid
import subprocess
import renderer
import importlib
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class CookieRotationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle cookie rotation on every request."""

    async def dispatch(self, request: Request, call_next):
        # Get current state BEFORE processing request
        cookie_value = request.cookies.get(COOKIE_NAME)
        key_ring = get_latched_keys()

        # Determine if we should set/rotate cookie
        should_set_cookie = False
        new_key = None

        if not key_ring:
            # No latch exists - create one and latch this device BEFORE processing
            logging.info("No key ring found. Creating initial key ring for device.")
            new_key = generate_secure_key()
            save_latched_key(new_key)
            should_set_cookie = True
            logging.info(f"Key ring created. First key saved to {KEY_FILE}")
            # Update request's cookie for this request so auth checks pass
            request._cookies[COOKIE_NAME] = new_key
        elif not cookie_value:
            # No cookie but ring exists - unauthorized device
            cookie_prefix = "none"
            logging.warning(f"Middleware: No cookie sent (key ring has {len(key_ring)} keys)")
        elif cookie_value in key_ring:
            # Cookie matches one of the keys - rotate to new key
            key_index = key_ring.index(cookie_value)
            cookie_prefix = cookie_value[:8]
            logging.info(f"Middleware: Cookie {cookie_prefix}... matched key #{key_index}, rotating")
            new_key = generate_secure_key()
            save_latched_key(new_key)
            should_set_cookie = True
            # Update request's cookie for this request
            request._cookies[COOKIE_NAME] = new_key
        else:
            # Cookie doesn't match any key in ring
            cookie_prefix = cookie_value[:8] if cookie_value else "none"
            logging.warning(f"Middleware: Cookie {cookie_prefix}... not in key ring (ring size: {len(key_ring)})")

        # Process the request (NOW the latch exists and cookie is set)
        response = await call_next(request)

        # Set cookie in response if needed
        if should_set_cookie and new_key:
            response.set_cookie(
                key=COOKIE_NAME,
                value=new_key,
                httponly=True,
                max_age=COOKIE_MAX_AGE,
                samesite="strict",
                secure=False  # Set to True if using HTTPS
            )

        return response

app.add_middleware(CookieRotationMiddleware)

DATA_FILE = "telemetry.json"
CHART_FILE = "chart.png"
SALT = "SALT"
LOCK = threading.Lock()

# Cookie-based authentication
KEY_DIR = "key"
KEY_FILE = os.path.join(KEY_DIR, "latched")
COOKIE_NAME = "auth_token"
COOKIE_MAX_AGE = 365 * 24 * 60 * 60  # 1 year in seconds

def generate_secure_key():
    """Generate a cryptographically secure random 64-character hex token."""
    return secrets.token_hex(32)

def get_latched_keys():
    """Read the key ring from disk. Returns list of valid keys (newest first)."""
    if not os.path.exists(KEY_FILE):
        return []
    try:
        with open(KEY_FILE, "r") as f:
            data = json.load(f)
            return data.get("keys", [])
    except (json.JSONDecodeError, KeyError):
        # Legacy format - single key as plain text
        with open(KEY_FILE, "r") as f:
            legacy_key = f.read().strip()
            if legacy_key:
                return [legacy_key]
        return []

def save_latched_key(new_key: str):
    """Save new key to the key ring, keeping last 5 keys."""
    try:
        os.makedirs(KEY_DIR, exist_ok=True)

        # Get existing keys
        existing_keys = get_latched_keys()

        # Add new key at front, keep only last 5
        key_ring = [new_key] + existing_keys
        key_ring = key_ring[:5]  # Keep max 5 keys

        # Save as JSON
        data = {
            "keys": key_ring,
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(KEY_FILE, "w") as f:
            json.dump(data, f, indent=2)

        logging.info(f"Saved key to ring (ring size: {len(key_ring)})")
    except Exception as e:
        logging.error(f"Failed to save latch key: {e}")
        raise

def verify_auth(request: Request):
    """Verify that the request has a valid auth cookie."""
    cookie_value = request.cookies.get(COOKIE_NAME)
    key_ring = get_latched_keys()

    if not key_ring:
        # No latch exists yet - deny (shouldn't happen with middleware)
        logging.warning("Auth check: No key ring exists")
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not cookie_value:
        logging.warning(f"Auth check: Missing cookie (key ring has {len(key_ring)} keys)")
        raise HTTPException(status_code=403, detail="Unauthorized")

    if cookie_value not in key_ring:
        cookie_prefix = cookie_value[:8] if cookie_value else "none"
        logging.warning(f"Auth check: Cookie {cookie_prefix}... not in key ring (ring size: {len(key_ring)})")
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Find which key matched
    key_index = key_ring.index(cookie_value)
    logging.info(f"Auth check: Cookie matched key #{key_index} in ring")

def get_sillykey():
    date_str = datetime.now().strftime("%Y-%m-%d")
    raw = f"{date_str}{SALT}"
    return hashlib.md5(raw.encode()).hexdigest()[-4:]

def verify_sillykey(sillykey: str):
    if sillykey != get_sillykey():
        raise HTTPException(status_code=403, detail="Invalid sillykey")

def git_sync(message: str):
    try:
        # Ensure we are up to date before pushing
        subprocess.run(["git", "pull", "--rebase", "origin", "master"], check=True)
        subprocess.run(["git", "add", DATA_FILE], check=True)
        # Only commit if there are changes
        status = subprocess.run(["git", "status", "--porcelain", DATA_FILE], capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(["git", "commit", "-m", f"telemetry: {message}"], check=True)
            subprocess.run(["git", "push", "origin", "master"], check=True)
    except Exception as e:
        print(f"Git sync failed: {e}")

def run_updates(new_data, message: str):
    with LOCK:
        with open(DATA_FILE, "w") as f:
            json.dump(new_data, f, indent=4)
        try:
            renderer.generate_chart(new_data, CHART_FILE)
        except Exception as e:
            print(f"Chart generation failed: {e}")
        git_sync(message)

def auto_pull_worker():
    """Background task to pull remote changes and refresh chart."""
    # Test comment: verifying auto-pull detection works
    while True:
        try:
            with LOCK:
                # Check for remote changes
                subprocess.run(["git", "fetch", "origin", "master"], check=True)
                res = subprocess.run(["git", "rev-list", "HEAD..origin/master", "--count"], capture_output=True, text=True)
                if res.stdout.strip() != "0":
                    logging.info("Remote changes detected. Synchronizing...")

                    # Get list of changed files
                    diff_result = subprocess.run(
                        ["git", "diff", "--name-only", "HEAD", "origin/master"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    changed_files = set(diff_result.stdout.strip().split('\n'))

                    subprocess.run(["git", "pull", "--rebase", "origin", "master"], check=True)

                    logging.info(f"Changed files: {changed_files}")

                    # Determine if only data files changed
                    data_only_files = {"telemetry.json", "chart.png", "server.log"}
                    non_data_changes = changed_files - data_only_files

                    if non_data_changes:
                        # Code changed - regenerate chart before restart
                        logging.info(f"Code changes detected: {non_data_changes}")

                        # Always regenerate chart on code changes
                        if os.path.exists(DATA_FILE):
                            logging.info("Regenerating chart with new code...")
                            with open(DATA_FILE, "r") as f:
                                data = json.load(f)
                            renderer.generate_chart(data, CHART_FILE)
                            logging.info("Chart regenerated with latest code.")

                        logging.info("Restarting service...")
                        subprocess.Popen(["sudo", "systemctl", "restart", "fasttrack"])
                        return  # Exit worker, service will restart with new code
                    else:
                        # Only data files changed - just regenerate chart
                        logging.info("Only data files changed. Regenerating chart...")
                        if os.path.exists(DATA_FILE):
                            with open(DATA_FILE, "r") as f:
                                data = json.load(f)
                            renderer.generate_chart(data, CHART_FILE)
                            logging.info("Chart regenerated.")
        except Exception as e:
            logging.error(f"Auto-pull worker error: {e}")
        time.sleep(60)

@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=auto_pull_worker, daemon=True)
    thread.start()

def update_data(new_data, message: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_updates, new_data, message)

@app.get("/api/telemetry")
@limiter.limit("20/5 minutes")
async def get_telemetry(
    request: Request,
    action: str = None,
    sillykey: str = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    # Add params
    timestamp: str = None,
    key: str = None,
    value: str = None,
    simulated: bool = False,
    # Batch params
    payload: str = None
):
    # READ operation
    if not action:
        if not os.path.exists(DATA_FILE):
            return []
        with open(DATA_FILE, "r") as f:
            return json.load(f)

    # WRITE operations (Auth required)
    verify_auth(request)

    if action == "add":
        if not key or value is None:
            raise HTTPException(status_code=400, detail="Missing key or value")
        
        ts = timestamp or datetime.utcnow().isoformat() + "Z"
        
        # Try to parse value as float if possible
        try:
            val = float(value)
        except ValueError:
            val = value

        entry = {
            "id": str(uuid.uuid4()),
            "key": key,
            "value": val,
            "simulated": simulated
        }

        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        
        # Find existing block for this timestamp or create new
        found = False
        for block in data:
            if block["timestamp"] == ts:
                block["entries"].append(entry)
                found = True
                break
        
        if not found:
            data.append({
                "timestamp": ts,
                "entries": [entry]
            })
        
        data.sort(key=lambda x: x["timestamp"])
        update_data(data, f"add {key}", background_tasks)
        return {"status": "success", "id": entry["id"]}

    elif action == "batch":
        if not payload:
            raise HTTPException(status_code=400, detail="Missing payload")
        try:
            new_data = json.loads(payload)
            update_data(new_data, "batch sync", background_tasks)
            return {"status": "success"}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

    else:
        raise HTTPException(status_code=400, detail="Invalid action")

@app.get("/api/config")
async def get_config():
    return {
        "sillykey": get_sillykey(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/api/latest")
async def get_latest():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    
    if not data:
        return {}
    
    # Extract most recent values for all keys
    latest = {}
    # Iterate backwards to find last occurrence of each key
    for block in reversed(data):
        for entry in block["entries"]:
            if entry["key"] not in latest:
                latest[entry["key"]] = entry["value"]
    return latest

@app.get("/")
async def read_index(request: Request):
    # Entry page - requires auth
    cookie_value = request.cookies.get(COOKIE_NAME)
    key_ring = get_latched_keys()

    if not key_ring or not cookie_value or cookie_value not in key_ring:
        # Not authorized - redirect to read-only graph page
        return RedirectResponse(url="/graph", status_code=302)

    return FileResponse("static/index.html")

@app.get("/meal")
async def read_meal(request: Request):
    # Entry page - requires auth
    cookie_value = request.cookies.get(COOKIE_NAME)
    key_ring = get_latched_keys()

    if not key_ring or not cookie_value or cookie_value not in key_ring:
        # Not authorized - redirect to read-only graph page
        return RedirectResponse(url="/graph", status_code=302)

    return FileResponse("static/meal.html")

@app.get("/graph")
async def read_graph():
    return FileResponse("static/graph.html")

@app.get("/api/graph")
async def get_pure_graph():
    if not os.path.exists(CHART_FILE):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            renderer.generate_chart(data, CHART_FILE)
        else:
            raise HTTPException(status_code=404, detail="Chart not found")
    # Disable caching - always fetch fresh chart
    return FileResponse(
        CHART_FILE,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/api/chart")
async def get_chart():
    if not os.path.exists(CHART_FILE):
        # Generate it if it doesn't exist
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            renderer.generate_chart(data, CHART_FILE)
        else:
            raise HTTPException(status_code=404, detail="Chart not found and no data available")
    # Disable caching - always fetch fresh chart
    return FileResponse(
        CHART_FILE,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
