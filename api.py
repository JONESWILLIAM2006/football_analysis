from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
import json
from datetime import datetime
from football_analysis import FootballAnalysisSystem
import redis
from pymongo import MongoClient

app = FastAPI(title="Football Analysis API", version="1.0.0")

# Initialize connections
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    mongo_client = MongoClient('mongodb://localhost:27017/')
    db = mongo_client.football_analysis
except:
    redis_client = None
    mongo_client = None

analysis_system = FootballAnalysisSystem()

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/process_video")
async def process_video(video_path: str, background_tasks: BackgroundTasks):
    job_id = f"job_{int(datetime.now().timestamp())}"
    
    if redis_client:
        redis_client.set(f"job:{job_id}", json.dumps({"status": "processing", "progress": 0}))
    
    background_tasks.add_task(analyze_video_background, video_path, job_id)
    
    return {"job_id": job_id, "status": "started"}

@app.get("/api/job_status/{job_id}")
def get_job_status(job_id: str):
    if redis_client:
        status = redis_client.get(f"job:{job_id}")
        if status:
            return json.loads(status)
    return {"status": "not_found"}

@app.websocket("/ws/live_analysis")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            # Mock real-time data
            data = {
                "timestamp": datetime.now().isoformat(),
                "events": [],
                "player_positions": {},
                "ball_position": [400, 300],
                "xg_live": 0.15,
                "possession": {"team_1": 0.6, "team_2": 0.4}
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(1/30)  # 30 FPS
            
        except Exception as e:
            break

async def analyze_video_background(video_path: str, job_id: str):
    try:
        results = analysis_system.process_video(video_path)
        
        if redis_client:
            redis_client.set(f"job:{job_id}", json.dumps({
                "status": "completed",
                "results": results
            }))
        
        if mongo_client:
            db.analysis_results.insert_one({
                "job_id": job_id,
                "results": results,
                "timestamp": datetime.now()
            })
            
    except Exception as e:
        if redis_client:
            redis_client.set(f"job:{job_id}", json.dumps({
                "status": "failed",
                "error": str(e)
            }))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)