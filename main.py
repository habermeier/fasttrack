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

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class CookieRotationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle cookie rotation on every request."""

    async def dispatch(self, request: Request, call_next):
        # Get current state
        cookie_value = request.cookies.get(COOKIE_NAME)
        latched_key = get_latched_key()

        # Process the request
        response = await call_next(request)

        # Determine if we should set/rotate cookie
        should_set_cookie = False
        new_key = None

        if not latched_key:
            # No latch exists - create one and latch this device
            new_key = generate_secure_key()
            save_latched_key(new_key)
            should_set_cookie = True
        elif cookie_value == latched_key:
            # Cookie matches - rotate to new key
            new_key = generate_secure_key()
            save_latched_key(new_key)
            should_set_cookie = True
        # else: cookie doesn't match - don't send cookie

        # Set cookie if needed
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

def get_latched_key():
    """Read the latched key from disk. Returns None if file doesn't exist."""
    if not os.path.exists(KEY_FILE):
        return None
    with open(KEY_FILE, "r") as f:
        return f.read().strip()

def save_latched_key(key: str):
    """Save the latched key to disk."""
    os.makedirs(KEY_DIR, exist_ok=True)
    with open(KEY_FILE, "w") as f:
        f.write(key)

def verify_auth(request: Request):
    """Verify that the request has a valid auth cookie."""
    cookie_value = request.cookies.get(COOKIE_NAME)
    latched_key = get_latched_key()

    if not latched_key:
        # No latch exists yet - deny (shouldn't happen with middleware)
        raise HTTPException(status_code=403, detail="Unauthorized")

    if cookie_value != latched_key:
        raise HTTPException(status_code=403, detail="Unauthorized")

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
    while True:
        try:
            with LOCK:
                # Check for remote changes
                subprocess.run(["git", "fetch", "origin", "master"], check=True)
                res = subprocess.run(["git", "rev-list", "HEAD..origin/master", "--count"], capture_output=True, text=True)
                if res.stdout.strip() != "0":
                    print(f"[{datetime.now()}] Remote changes detected. Synchronizing...")

                    # Track hashes of important files
                    files_to_watch = ["telemetry.json", "renderer.py", "main.py"]
                    hashes_before = {}
                    for f_path in files_to_watch:
                        if os.path.exists(f_path):
                            with open(f_path, "rb") as f:
                                hashes_before[f_path] = hashlib.md5(f.read()).hexdigest()

                    subprocess.run(["git", "pull", "--rebase", "origin", "master"], check=True)

                    # Check hashes after pull
                    changed = []
                    for f_path in files_to_watch:
                        if os.path.exists(f_path):
                            with open(f_path, "rb") as f:
                                h_after = hashlib.md5(f.read()).hexdigest()
                                if h_after != hashes_before.get(f_path):
                                    changed.append(f_path)

                    # If main.py changed, restart the service
                    if "main.py" in changed:
                        print(f"[{datetime.now()}] main.py changed. Restarting service...")
                        subprocess.Popen(["sudo", "systemctl", "restart", "fasttrack"])
                        return  # Exit worker, service will restart with new code

                    if "renderer.py" in changed:
                        print(f"[{datetime.now()}] renderer.py changed. Hot-reloading module...")
                        importlib.reload(renderer)

                    if "telemetry.json" in changed or "renderer.py" in changed:
                        if os.path.exists(DATA_FILE):
                            with open(DATA_FILE, "r") as f:
                                data = json.load(f)
                            renderer.generate_chart(data, CHART_FILE)
                            print(f"[{datetime.now()}] Chart regenerated after updates.")
                    else:
                        print(f"[{datetime.now()}] Relevant files unchanged. Skipping updates.")
        except Exception as e:
            print(f"Auto-pull worker error: {e}")
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
    latched_key = get_latched_key()

    if not latched_key or cookie_value != latched_key:
        # Not authorized - redirect to read-only graph page
        return RedirectResponse(url="/graph", status_code=302)

    return FileResponse("static/index.html")

@app.get("/meal")
async def read_meal(request: Request):
    # Entry page - requires auth
    cookie_value = request.cookies.get(COOKIE_NAME)
    latched_key = get_latched_key()

    if not latched_key or cookie_value != latched_key:
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
    return FileResponse(CHART_FILE)

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
    return FileResponse(CHART_FILE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
