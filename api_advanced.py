"""
Advanced Football Analysis API
FastAPI backend for real-time processing and WebSocket streaming
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import cv2
import numpy as np
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import redis
from pymongo import MongoClient
import gridfs
from minio import Minio
import io
import base64
from pathlib import Path

# Import our analysis pipeline
from football_analysis import AdvancedFootballAnalysis

app = FastAPI(title="Advanced Football Analysis API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connections
redis_client = redis.Redis(host='redis', port=6379, db=0)
mongo_client = MongoClient('mongodb://mongo:27017/')
db = mongo_client.football_analysis
fs = gridfs.GridFS(db)

# MinIO client
minio_client = Minio(
    'minio:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

# Global analysis pipeline
analyzer = AdvancedFootballAnalysis()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.analysis_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.analysis_sessions[session_id] = {
            'websocket': websocket,
            'status': 'connected',
            'start_time': datetime.now()
        }

    def disconnect(self, websocket: WebSocket, session_id: str):
        self.active_connections.remove(websocket)
        if session_id in self.analysis_sessions:
            del self.analysis_sessions[session_id]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_text(json.dumps(message))

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Advanced Football Analysis API", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """System health check"""
    status = {
        "api": "healthy",
        "redis": "unknown",
        "mongodb": "unknown",
        "minio": "unknown",
        "gpu": "unknown"
    }
    
    # Check Redis
    try:
        redis_client.ping()
        status["redis"] = "healthy"
    except:
        status["redis"] = "unhealthy"
    
    # Check MongoDB
    try:
        mongo_client.admin.command('ping')
        status["mongodb"] = "healthy"
    except:
        status["mongodb"] = "unhealthy"
    
    # Check MinIO
    try:
        minio_client.list_buckets()
        status["minio"] = "healthy"
    except:
        status["minio"] = "unhealthy"
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            status["gpu"] = f"healthy - {torch.cuda.get_device_name(0)}"
        else:
            status["gpu"] = "no_gpu"
    except:
        status["gpu"] = "unknown"
    
    return status

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video for analysis"""
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        
        # Save to MinIO
        file_data = await file.read()
        minio_client.put_object(
            "videos",
            filename,
            io.BytesIO(file_data),
            len(file_data),
            content_type=file.content_type
        )
        
        # Store metadata in MongoDB
        metadata = {
            "file_id": file_id,
            "filename": filename,
            "original_name": file.filename,
            "size": len(file_data),
            "content_type": file.content_type,
            "upload_time": datetime.now(),
            "status": "uploaded"
        }
        
        db.videos.insert_one(metadata)
        
        return {"file_id": file_id, "status": "uploaded", "filename": filename}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/start_analysis/{file_id}")
async def start_analysis(file_id: str, background_tasks: BackgroundTasks):
    """Start video analysis in background"""
    try:
        # Get video metadata
        video_meta = db.videos.find_one({"file_id": file_id})
        if not video_meta:
            return JSONResponse(
                status_code=404,
                content={"error": "Video not found"}
            )
        
        # Create analysis job
        job_id = str(uuid.uuid4())\n        job_data = {\n            "job_id": job_id,\n            "file_id": file_id,\n            "status": "queued",\n            "created_at": datetime.now(),\n            "progress": 0\n        }\n        \n        db.analysis_jobs.insert_one(job_data)\n        \n        # Add to background tasks\n        background_tasks.add_task(process_video_analysis, file_id, job_id)\n        \n        return {"job_id": job_id, "status": "queued"}\n        \n    except Exception as e:\n        return JSONResponse(\n            status_code=500,\n            content={"error": str(e)}\n        )\n\n@app.get("/api/job_status/{job_id}")\nasync def get_job_status(job_id: str):\n    """Get analysis job status"""\n    try:\n        job = db.analysis_jobs.find_one({"job_id": job_id})\n        if not job:\n            return JSONResponse(\n                status_code=404,\n                content={"error": "Job not found"}\n            )\n        \n        # Convert ObjectId to string for JSON serialization\n        job["_id"] = str(job["_id"])\n        \n        return job\n        \n    except Exception as e:\n        return JSONResponse(\n            status_code=500,\n            content={"error": str(e)}\n        )\n\n@app.get("/api/results/{job_id}")\nasync def get_analysis_results(job_id: str):\n    """Get analysis results"""\n    try:\n        results = db.analysis_results.find_one({"job_id": job_id})\n        if not results:\n            return JSONResponse(\n                status_code=404,\n                content={"error": "Results not found"}\n            )\n        \n        # Convert ObjectId to string\n        results["_id"] = str(results["_id"])\n        \n        return results\n        \n    except Exception as e:\n        return JSONResponse(\n            status_code=500,\n            content={"error": str(e)}\n        )\n\n@app.websocket("/ws/live_analysis/{session_id}")\nasync def websocket_live_analysis(websocket: WebSocket, session_id: str):\n    """WebSocket endpoint for live analysis"""\n    await manager.connect(websocket, session_id)\n    \n    try:\n        while True:\n            # Receive frame data\n            data = await websocket.receive_text()\n            frame_data = json.loads(data)\n            \n            if frame_data.get("type") == "frame":\n                # Decode base64 frame\n                frame_bytes = base64.b64decode(frame_data["frame"])\n                nparr = np.frombuffer(frame_bytes, np.uint8)\n                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)\n                \n                # Process frame\n                frame_idx = frame_data.get("frame_idx", 0)\n                results = analyzer.process_frame(frame, frame_idx)\n                \n                # Send results back\n                response = {\n                    "type": "analysis_result",\n                    "frame_idx": frame_idx,\n                    "results": results,\n                    "timestamp": datetime.now().isoformat()\n                }\n                \n                await manager.send_personal_message(response, websocket)\n                \n    except WebSocketDisconnect:\n        manager.disconnect(websocket, session_id)\n    except Exception as e:\n        await manager.send_personal_message(\n            {"type": "error", "message": str(e)}, \n            websocket\n        )\n        manager.disconnect(websocket, session_id)\n\n@app.post("/api/rtsp_stream")\nasync def add_rtsp_stream(stream_data: dict):\n    """Add RTSP stream for live analysis"""\n    try:\n        stream_id = str(uuid.uuid4())\n        stream_url = stream_data.get("url")\n        stream_name = stream_data.get("name", "Unknown Stream")\n        \n        # Store stream info\n        stream_info = {\n            "stream_id": stream_id,\n            "url": stream_url,\n            "name": stream_name,\n            "status": "active",\n            "created_at": datetime.now()\n        }\n        \n        db.rtsp_streams.insert_one(stream_info)\n        \n        # Add to Redis for worker processing\n        redis_client.lpush("rtsp_streams", json.dumps(stream_info))\n        \n        return {"stream_id": stream_id, "status": "added"}\n        \n    except Exception as e:\n        return JSONResponse(\n            status_code=500,\n            content={"error": str(e)}\n        )\n\n@app.get("/api/streams")\nasync def get_active_streams():\n    """Get all active RTSP streams"""\n    try:\n        streams = list(db.rtsp_streams.find({"status": "active"}))\n        \n        # Convert ObjectIds to strings\n        for stream in streams:\n            stream["_id"] = str(stream["_id"])\n            \n        return {"streams": streams}\n        \n    except Exception as e:\n        return JSONResponse(\n            status_code=500,\n            content={"error": str(e)}\n        )\n\n@app.post("/api/semantic_search")\nasync def semantic_search(query_data: dict):\n    """Search events using natural language"""\n    try:\n        query = query_data.get("query", "")\n        match_id = query_data.get("match_id")\n        \n        # Simple keyword-based search (would use embeddings in production)\n        keywords = query.lower().split()\n        \n        # Search in events\n        events = list(db.events.find({\n            "match_id": match_id,\n            "$or": [\n                {"type": {"$in": keywords}},\n                {"description": {"$regex": "|".join(keywords), "$options": "i"}}\n            ]\n        }).limit(20))\n        \n        # Convert ObjectIds\n        for event in events:\n            event["_id"] = str(event["_id"])\n            \n        return {"results": events, "query": query}\n        \n    except Exception as e:\n        return JSONResponse(\n            status_code=500,\n            content={"error": str(e)}\n        )\n\n@app.get("/api/match_stats/{match_id}")\nasync def get_match_stats(match_id: str):\n    """Get comprehensive match statistics"""\n    try:\n        # Get match data\n        match = db.matches.find_one({"match_id": match_id})\n        if not match:\n            return JSONResponse(\n                status_code=404,\n                content={"error": "Match not found"}\n            )\n        \n        # Get events\n        events = list(db.events.find({"match_id": match_id}))\n        \n        # Calculate statistics\n        stats = {\n            "match_id": match_id,\n            "total_events": len(events),\n            "possession": calculate_possession(events),\n            "shots": count_events_by_type(events, "shot"),\n            "passes": count_events_by_type(events, "pass"),\n            "fouls": count_events_by_type(events, "foul"),\n            "xg": calculate_xg(events)\n        }\n        \n        return stats\n        \n    except Exception as e:\n        return JSONResponse(\n            status_code=500,\n            content={"error": str(e)}\n        )\n\n# Background task functions\nasync def process_video_analysis(file_id: str, job_id: str):\n    """Background task to process video analysis"""\n    try:\n        # Update job status\n        db.analysis_jobs.update_one(\n            {"job_id": job_id},\n            {"$set": {"status": "processing", "started_at": datetime.now()}}\n        )\n        \n        # Get video from MinIO\n        video_meta = db.videos.find_one({"file_id": file_id})\n        filename = video_meta["filename"]\n        \n        response = minio_client.get_object("videos", filename)\n        video_data = response.read()\n        \n        # Save temporarily\n        temp_path = f"/tmp/{filename}"\n        with open(temp_path, "wb") as f:\n            f.write(video_data)\n        \n        # Process video\n        cap = cv2.VideoCapture(temp_path)\n        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n        \n        results = {\n            "events": [],\n            "var_decisions": [],\n            "tactical_insights": [],\n            "player_stats": {},\n            "match_stats": {}\n        }\n        \n        prev_frame = None\n        for frame_idx in range(0, frame_count, 5):  # Process every 5th frame\n            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)\n            ret, frame = cap.read()\n            \n            if not ret:\n                break\n            \n            # Process frame\n            frame_data = analyzer.process_frame(frame, frame_idx, prev_frame)\n            prev_frame = frame\n            \n            # Update progress\n            progress = int((frame_idx / frame_count) * 100)\n            db.analysis_jobs.update_one(\n                {"job_id": job_id},\n                {"$set": {"progress": progress}}\n            )\n        \n        cap.release()\n        \n        # Combine results\n        results.update(analyzer.analysis_results)\n        \n        # Save results\n        results["job_id"] = job_id\n        results["file_id"] = file_id\n        results["completed_at"] = datetime.now()\n        \n        db.analysis_results.insert_one(results)\n        \n        # Update job status\n        db.analysis_jobs.update_one(\n            {"job_id": job_id},\n            {"$set": {"status": "completed", "completed_at": datetime.now(), "progress": 100}}\n        )\n        \n        # Clean up\n        Path(temp_path).unlink(missing_ok=True)\n        \n    except Exception as e:\n        # Update job with error\n        db.analysis_jobs.update_one(\n            {"job_id": job_id},\n            {"$set": {"status": "failed", "error": str(e), "failed_at": datetime.now()}}\n        )\n\n# Utility functions\ndef calculate_possession(events: List[Dict]) -> Dict:\n    """Calculate possession statistics"""\n    team_a_events = len([e for e in events if e.get("team") == "team_A"])\n    team_b_events = len([e for e in events if e.get("team") == "team_B"])\n    total = team_a_events + team_b_events\n    \n    if total == 0:\n        return {"team_A": 50, "team_B": 50}\n    \n    return {\n        "team_A": round((team_a_events / total) * 100, 1),\n        "team_B": round((team_b_events / total) * 100, 1)\n    }\n\ndef count_events_by_type(events: List[Dict], event_type: str) -> Dict:\n    """Count events by type for each team"""\n    team_a_count = len([e for e in events if e.get("type") == event_type and e.get("team") == "team_A"])\n    team_b_count = len([e for e in events if e.get("type") == event_type and e.get("team") == "team_B"])\n    \n    return {"team_A": team_a_count, "team_B": team_b_count}\n\ndef calculate_xg(events: List[Dict]) -> Dict:\n    """Calculate expected goals"""\n    team_a_xg = sum([e.get("xg", 0) for e in events if e.get("type") == "shot" and e.get("team") == "team_A"])\n    team_b_xg = sum([e.get("xg", 0) for e in events if e.get("type") == "shot" and e.get("team") == "team_B"])\n    \n    return {\n        "team_A": round(team_a_xg, 2),\n        "team_B": round(team_b_xg, 2)\n    }\n\nif __name__ == "__main__":\n    import uvicorn\n    uvicorn.run(app, host="0.0.0.0", port=8000)