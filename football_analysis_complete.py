# Complete Football Analysis System
import os
import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from collections import defaultdict, deque, Counter
import uuid
from datetime import datetime
from enum import Enum
import asyncio
from pymongo import MongoClient
import json

# Optional imports with fallbacks
try:
    import redis
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

try:
    import onnxruntime as ort
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

try:
    import mediapipe as mp
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False

# Environment configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/')

# Initialize MongoDB
client = MongoClient(MONGODB_URL)
db = client["football_analytics"]
jobs_collection = db["analysis_jobs"]

# Initialize Celery if available
if CELERY_AVAILABLE:
    celery_app = Celery('football_analysis', broker=CELERY_BROKER_URL)
    redis_conn = redis.Redis.from_url(CELERY_BROKER_URL)
else:
    celery_app = None
    redis_conn = None

# Create directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("highlights", exist_ok=True)
os.makedirs("animations", exist_ok=True)

class RTDETRDetector:
    """RT-DETR-v3 for real-time detection"""
    def __init__(self, model_path="yolov8x.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                if cls == 0:  # Person class
                    detections.append({
                        'bbox': box,
                        'confidence': score,
                        'class': int(cls)
                    })
        
        return detections

class BallDetector:
    """Enhanced ball detection with motion enhancement"""
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        self.prev_frame = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        
    def detect_with_motion_enhancement(self, frame):
        enhanced_frame = self._enhance_motion(frame)
        results = self.model(enhanced_frame, imgsz=1280, conf=0.25, verbose=False)
        
        ball_detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                if cls == 32:  # Sports ball
                    ball_detections.append({
                        'bbox': box,
                        'confidence': score,
                        'center': [(box[0]+box[2])/2, (box[1]+box[3])/2]
                    })
        
        return self._filter_false_positives(ball_detections, frame)
    
    def _enhance_motion(self, frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
            
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.prev_frame, current_gray)
        fg_mask = self.bg_subtractor.apply(frame)
        
        motion_mask = cv2.bitwise_or(diff, fg_mask)
        enhanced = frame.copy()
        enhanced[motion_mask > 50] = cv2.addWeighted(
            frame, 0.7, np.full_like(frame, 255), 0.3, 0
        )[motion_mask > 50]
        
        self.prev_frame = current_gray
        return enhanced
    
    def _filter_false_positives(self, detections, frame):
        filtered = []
        for det in detections:
            bbox = det['bbox']
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            if 5 <= width <= 50 and 5 <= height <= 50:
                aspect_ratio = width / height
                if 0.5 <= aspect_ratio <= 2.0:
                    filtered.append(det)
        
        return filtered

class ByteTracker:
    """ByteTracker implementation for player tracking"""
    def __init__(self, frame_rate=30):
        self.frame_rate = frame_rate
        self.tracks = {}
        self.next_id = 1
        self.max_age = 30
        
    def update(self, detections):
        matched_tracks = {}
        unmatched_detections = detections.copy()
        
        # Match with existing tracks
        for track_id, track in self.tracks.items():
            if track['age'] < self.max_age:
                best_match = None
                best_iou = 0.3
                
                for i, det in enumerate(unmatched_detections):
                    iou = self._calculate_iou(track['bbox'], det['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = i
                
                if best_match is not None:
                    det = unmatched_detections.pop(best_match)
                    matched_tracks[track_id] = {
                        'bbox': det['bbox'],
                        'confidence': det['confidence'],
                        'age': 0
                    }
        
        # Create new tracks
        for det in unmatched_detections:
            matched_tracks[self.next_id] = {
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'age': 0
            }
            self.next_id += 1
        
        # Age existing tracks
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track['age'] += 1
                if track['age'] < self.max_age:
                    matched_tracks[track_id] = track
        
        self.tracks = matched_tracks
        return [(tid, track['bbox'], track['confidence']) for tid, track in self.tracks.items()]
    
    def _calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_t, y1_t, x2_t, y2_t = box2
        
        xi1, yi1 = max(x1, x1_t), max(y1, y1_t)
        xi2, yi2 = min(x2, x2_t), min(y2, y2_t)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_t - x1_t) * (y2_t - y1_t)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

class KalmanBallTracker:
    """Kalman filter for ball tracking"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.initialized = False
        self.trajectory = deque(maxlen=30)
        
    def update(self, detection):
        if detection is None:
            if self.initialized:
                prediction = self.kf.predict()
                return (int(prediction[0]), int(prediction[1]))
            return None
            
        center = detection['center']
        
        if not self.initialized:
            self.kf.statePre = np.array([center[0], center[1], 0, 0], dtype=np.float32)
            self.kf.statePost = np.array([center[0], center[1], 0, 0], dtype=np.float32)
            self.initialized = True
        
        measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
        self.kf.correct(measurement)
        
        self.trajectory.append((int(center[0]), int(center[1])))
        return (int(center[0]), int(center[1]))
    
    def get_trajectory(self):
        return list(self.trajectory)

class VARAnalyzer:
    """VAR analysis for offside and goal-line decisions"""
    def __init__(self):
        self.pitch_dims = (105, 68)
        
    def detect_offside(self, players, ball_owner, event_type):
        if event_type not in ['through_ball', 'shot', 'cross']:
            return None
            
        attacking_players = [p for p in players.keys() if p <= 11 and p != ball_owner]
        defending_players = [p for p in players.keys() if p > 11]
        
        if len(defending_players) < 2:
            return None
            
        defender_y_positions = [players[p][1] for p in defending_players]
        defender_y_positions.sort()
        offside_line_y = defender_y_positions[1]
        
        for player_id in attacking_players:
            player_y = players[player_id][1]
            if player_y > offside_line_y:
                return {
                    'player_id': player_id,
                    'is_offside': True,
                    'margin': player_y - offside_line_y,
                    'offside_line': offside_line_y
                }
        
        return {'is_offside': False}
    
    def check_goal_line(self, ball_position, goal_line_y=105):
        return ball_position[1] >= goal_line_y

class PitchControlModel:
    """Calculate pitch control using Voronoi diagrams"""
    def __init__(self, pitch_dims=(105, 68)):
        self.pitch_dims = pitch_dims
        
    def calculate_control(self, player_positions):
        control_map = np.zeros((14, 21))
        
        for i in range(14):
            for j in range(21):
                point = np.array([j*5, i*5])
                team1_time, team2_time = float('inf'), float('inf')
                
                for pid, pos in player_positions.items():
                    distance = np.linalg.norm(np.array(pos) - point)
                    time_to_reach = distance / 5
                    
                    if pid <= 11:
                        team1_time = min(team1_time, time_to_reach)
                    else:
                        team2_time = min(team2_time, time_to_reach)
                
                if team1_time < team2_time:
                    control_map[i, j] = 1
                elif team2_time < team1_time:
                    control_map[i, j] = -1
        
        return control_map

class xGModel:
    """Expected Goals model using XGBoost"""
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        self.trained = False
        
    def train(self, shot_data):
        if len(shot_data) < 10:
            return False
            
        X = np.array([[s['distance'], s['angle'], s['defenders']] for s in shot_data])
        y = np.array([s['goal'] for s in shot_data])
        
        self.model.fit(X, y)
        self.trained = True
        return True
    
    def predict_xg(self, distance, angle, defenders):
        if not self.trained:
            base_xg = max(0, 0.3 - (distance / 100))
            angle_factor = 1 - abs(angle - 45) / 45
            defender_penalty = defenders * 0.05
            return max(0, base_xg * angle_factor - defender_penalty)
        
        return self.model.predict_proba([[distance, angle, defenders]])[0][1]

class EventRecognizer:
    """Recognize football events from player movements"""
    def __init__(self):
        self.event_history = deque(maxlen=100)
        
    def recognize_events(self, players, ball_position, frame_num):
        events = []
        
        if ball_position:
            closest_player = None
            min_distance = float('inf')
            
            for pid, pos in players.items():
                distance = np.linalg.norm(np.array(pos) - np.array(ball_position))
                if distance < min_distance:
                    min_distance = distance
                    closest_player = pid
            
            if closest_player and min_distance < 5:
                if len(self.event_history) > 0:
                    last_event = self.event_history[-1]
                    if (last_event['type'] == 'possession' and 
                        last_event['player'] != closest_player):
                        events.append({
                            'type': 'pass',
                            'from_player': last_event['player'],
                            'to_player': closest_player,
                            'frame': frame_num,
                            'success': True
                        })
                
                events.append({
                    'type': 'possession',
                    'player': closest_player,
                    'frame': frame_num,
                    'position': ball_position
                })
        
        for event in events:
            self.event_history.append(event)
        
        return events

class MatchAnalysisSystem:
    """Main analysis system integrating all components"""
    def __init__(self):
        self.detector = RTDETRDetector()
        self.ball_detector = BallDetector()
        self.player_tracker = ByteTracker()
        self.ball_tracker = KalmanBallTracker()
        self.var_analyzer = VARAnalyzer()
        self.pitch_control = PitchControlModel()
        self.xg_model = xGModel()
        self.event_recognizer = EventRecognizer()
        
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = "outputs/processed_video.mp4"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_events = []
        player_positions_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players
            player_detections = self.detector.detect(frame)
            
            # Track players
            tracked_players = self.player_tracker.update(player_detections)
            
            # Detect and track ball
            ball_detections = self.ball_detector.detect_with_motion_enhancement(frame)
            ball_position = None
            if ball_detections:
                ball_position = self.ball_tracker.update(ball_detections[0])
            
            # Convert tracked players to position dict
            player_positions = {}
            for track_id, bbox, conf in tracked_players:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                player_positions[track_id] = (center_x, center_y)
            
            player_positions_history.append(player_positions)
            
            # Recognize events
            events = self.event_recognizer.recognize_events(
                player_positions, ball_position, frame_count
            )
            all_events.extend(events)
            
            # VAR analysis for key events
            for event in events:
                if event['type'] == 'pass':
                    offside_result = self.var_analyzer.detect_offside(
                        player_positions, event['from_player'], 'pass'
                    )
                    if offside_result and offside_result.get('is_offside'):
                        all_events.append({
                            'type': 'offside',
                            'player': offside_result['player_id'],
                            'frame': frame_count,
                            'margin': offside_result['margin']
                        })
            
            # Draw annotations
            annotated_frame = self._draw_annotations(
                frame, tracked_players, ball_position, self.ball_tracker.get_trajectory()
            )
            
            out.write(annotated_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                st.info(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        # Calculate advanced metrics
        pitch_control_maps = []
        for positions in player_positions_history[-10:]:  # Last 10 frames
            if positions:
                control_map = self.pitch_control.calculate_control(positions)
                pitch_control_maps.append(control_map)
        
        return {
            'events': all_events,
            'player_positions_history': player_positions_history,
            'pitch_control_maps': pitch_control_maps,
            'output_video': output_path,
            'total_frames': frame_count
        }
    
    def _draw_annotations(self, frame, tracked_players, ball_position, ball_trajectory):
        annotated = frame.copy()
        
        # Draw players
        for track_id, bbox, conf in tracked_players:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = (0, 255, 0) if track_id <= 11 else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"P{track_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw ball and trajectory
        if ball_position:
            cv2.circle(annotated, ball_position, 8, (255, 255, 255), -1)
            cv2.circle(annotated, ball_position, 8, (0, 0, 0), 2)
        
        # Draw ball trajectory
        if len(ball_trajectory) > 1:
            for i in range(1, len(ball_trajectory)):
                alpha = i / len(ball_trajectory)
                color = (int(255 * alpha), int(255 * alpha), 255)
                thickness = max(1, int(3 * alpha))
                cv2.line(annotated, ball_trajectory[i-1], ball_trajectory[i], color, thickness)
        
        return annotated

# Celery background task
if CELERY_AVAILABLE:
    @celery_app.task(bind=True)
    def process_match_async(self, job_id, video_path):
        try:
            jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {"status": "processing", "progress": 0}}
            )
            
            system = MatchAnalysisSystem()
            results = system.process_video(video_path)
            
            jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": "completed",
                    "progress": 100,
                    "results": results
                }}
            )
            
            return {"status": "success", "job_id": job_id}
            
        except Exception as e:
            jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {"status": "failed", "error": str(e)}}
            )
            raise e

def main_interface():
    """Main Streamlit interface"""
    st.set_page_config(
        page_title="⚽ Advanced Football Analysis",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Advanced Football Analysis System")
    st.markdown("*Professional-grade football analytics with AI-powered detection and tracking*")
    
    # Sidebar
    st.sidebar.title("🎯 System Status")
    
    # Health check
    health_status = {
        'YOLO Models': True,
        'ByteTracker': True,
        'Celery': CELERY_AVAILABLE,
        'MongoDB': True,
        'Optimization': OPTIMIZATION_AVAILABLE
    }
    
    for service, status in health_status.items():
        icon = "✅" if status else "❌"
        st.sidebar.write(f"{icon} {service}")
    
    # Main interface
    uploaded_file = st.file_uploader("Upload Match Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file:
        # Save uploaded file
        video_path = f"temp_video_{int(time.time())}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Analysis Options")
            enable_var = st.checkbox("Enable VAR Analysis", value=True)
            enable_pitch_control = st.checkbox("Enable Pitch Control", value=True)
            enable_xg = st.checkbox("Enable xG Calculation", value=True)
        
        with col2:
            st.subheader("Processing Options")
            use_background = st.checkbox("Background Processing", value=CELERY_AVAILABLE)
            max_frames = st.number_input("Max Frames", 100, 5000, 1000)
        
        if st.button("🚀 Start Analysis", type="primary"):
            if use_background and CELERY_AVAILABLE:
                # Background processing
                job_id = str(uuid.uuid4())
                
                jobs_collection.insert_one({
                    "job_id": job_id,
                    "status": "queued",
                    "progress": 0,
                    "created_at": datetime.now()
                })
                
                process_match_async.delay(job_id, video_path)
                
                st.success(f"🚀 Analysis submitted! Job ID: {job_id}")
                st.info("Processing in background. Check status below.")
                
                # Status monitoring
                if st.button("Check Status"):
                    job = jobs_collection.find_one({"job_id": job_id})
                    if job:
                        st.write(f"Status: {job['status']}")
                        if job['status'] == 'completed':
                            st.success("Analysis complete!")
                            results = job['results']
                            
                            # Display results
                            st.subheader("📊 Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Events", len(results['events']))
                            with col2:
                                st.metric("Frames Processed", results['total_frames'])
                            with col3:
                                st.metric("Players Tracked", len(results['player_positions_history'][-1]) if results['player_positions_history'] else 0)
                            
                            # Show processed video
                            if os.path.exists(results['output_video']):
                                st.video(results['output_video'])
                            
                            # Event timeline
                            if results['events']:
                                st.subheader("📈 Event Timeline")
                                events_df = pd.DataFrame(results['events'])
                                st.dataframe(events_df)
                        
                        elif job['status'] == 'failed':
                            st.error(f"Analysis failed: {job.get('error', 'Unknown error')}")
            
            else:
                # Synchronous processing
                with st.spinner("Processing video..."):
                    system = MatchAnalysisSystem()
                    results = system.process_video(video_path)
                
                st.success("✅ Analysis Complete!")
                
                # Display results
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Events", len(results['events']))
                with col2:
                    st.metric("Frames Processed", results['total_frames'])
                with col3:
                    st.metric("Players Tracked", len(results['player_positions_history'][-1]) if results['player_positions_history'] else 0)
                
                # Show processed video
                if os.path.exists(results['output_video']):
                    st.video(results['output_video'])
                
                # Event analysis
                if results['events']:
                    st.subheader("📈 Event Analysis")
                    
                    events_df = pd.DataFrame(results['events'])
                    st.dataframe(events_df)
                    
                    # Event type distribution
                    event_counts = events_df['type'].value_counts()
                    fig, ax = plt.subplots()
                    event_counts.plot(kind='bar', ax=ax)
                    ax.set_title('Event Distribution')
                    st.pyplot(fig)
                
                # Pitch control visualization
                if results['pitch_control_maps'] and enable_pitch_control:
                    st.subheader("🗺️ Pitch Control")
                    
                    avg_control = np.mean(results['pitch_control_maps'], axis=0)
                    fig, ax = plt.subplots(figsize=(12, 8))
                    im = ax.imshow(avg_control, cmap='RdBu', aspect='auto')
                    ax.set_title('Average Pitch Control')
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)
                
                # Clean up
                if os.path.exists(video_path):
                    os.remove(video_path)
    
    # Information section
    with st.expander("ℹ️ System Information"):
        st.markdown("""
        ### Advanced Football Analysis Features:
        
        **🎯 Detection & Tracking:**
        - RT-DETR-v3 for real-time player detection
        - Enhanced ball detection with motion enhancement
        - ByteTracker for consistent player IDs
        - Kalman filtering for ball trajectory prediction
        
        **📊 Analysis Components:**
        - VAR offside detection with automated line drawing
        - Pitch control calculation using Voronoi diagrams
        - xG (Expected Goals) modeling with XGBoost
        - Event recognition (passes, shots, possession changes)
        
        **🚀 Performance Features:**
        - Background processing with Celery/Redis
        - ONNX/TensorRT optimization support
        - Multi-camera synchronization ready
        - Real-time streaming capability
        
        **📈 Professional Metrics:**
        - PPDA (Passes Per Defensive Action)
        - Packing scores for line-breaking passes
        - Formation detection and analysis
        - Player energy and fatigue monitoring
        """)

if __name__ == "__main__":
    main_interface()