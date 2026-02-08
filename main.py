from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import json
import os
import hashlib
from datetime import datetime
import uuid
import subprocess
from renderer import generate_chart
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import threading

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

DATA_FILE = "telemetry.json"
CHART_FILE = "chart.png"
SALT = "SALT"
LOCK = threading.Lock()

def get_sillykey():
    date_str = datetime.now().strftime("%Y-%m-%d")
    raw = f"{date_str}{SALT}"
    return hashlib.md5(raw.encode()).hexdigest()[-4:]

def verify_sillykey(sillykey: str):
    if sillykey != get_sillykey():
        raise HTTPException(status_code=403, detail="Invalid sillykey")

def git_sync(message: str):
    try:
        subprocess.run(["git", "add", DATA_FILE], check=True)
        subprocess.run(["git", "commit", "-m", f"telemetry: {message}"], check=True)
        subprocess.run(["git", "push", "origin", "master"], check=True)
    except Exception as e:
        print(f"Git sync failed: {e}")

def update_data(new_data, message: str, background_tasks: BackgroundTasks):
    with LOCK:
        with open(DATA_FILE, "w") as f:
            json.dump(new_data, f, indent=4)
        generate_chart(new_data, CHART_FILE)
    background_tasks.add_task(git_sync, message)

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
    verify_sillykey(sillykey)

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
async def read_index():
    return FileResponse("static/index.html")

@app.get("/meal")
async def read_meal():
    return FileResponse("static/meal.html")

@app.get("/graph")
async def read_graph():
    return FileResponse("static/graph.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/api/chart")
async def get_chart():
    if not os.path.exists(CHART_FILE):
        # Generate it if it doesn't exist
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            generate_chart(data, CHART_FILE)
        else:
            raise HTTPException(status_code=404, detail="Chart not found and no data available")
    return FileResponse(CHART_FILE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
