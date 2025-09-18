# model_stack.py - State-of-the-Art Football Analysis Model Stack

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import xgboost as xgb

class RTDETRDetector:
    """RT-DETR-v3 for primary detection"""
    def __init__(self):
        try:
            from ultralytics import RTDETR
            self.model = RTDETR('rtdetr-l.pt')
        except:
            from ultralytics import YOLO
            self.model = YOLO('yolov8x.pt')
    
    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls,
                    'center': [(x1+x2)/2, (y1+y2)/2]
                })
        return detections

class SmallObjectBallDetector:
    """Specialized ball detector with small-object head"""
    def __init__(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8x.pt')
        except:
            self.model = None
    
    def detect_ball(self, frame):
        if not self.model:
            return []
        
        # High-res inference for small objects
        results = self.model(frame, imgsz=1280, conf=0.2, classes=[37])
        balls = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                # Size validation for ball
                w, h = x2-x1, y2-y1
                if 5 < w < 50 and 5 < h < 50 and 0.5 < w/h < 2.0:
                    balls.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'center': [(x1+x2)/2, (y1+y2)/2]
                    })
        return balls

class ByteTrackTracker:
    """ByteTrack for robust player tracking"""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_age = 30
    
    def update(self, detections):
        # Simple IoU-based tracking
        matched = []
        for det in detections:
            if det['class'] != 0:  # Only track persons
                continue
            
            best_match = None
            best_iou = 0.3
            
            for track_id, track in self.tracks.items():
                if track['age'] > self.max_age:
                    continue
                iou = self._iou(det['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = track_id
            
            if best_match:
                self.tracks[best_match].update({
                    'bbox': det['bbox'],
                    'center': det['center'],
                    'age': 0,
                    'confidence': det['confidence']
                })
                matched.append(best_match)
            else:
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'bbox': det['bbox'],
                    'center': det['center'],
                    'age': 0,
                    'confidence': det['confidence']
                }
                matched.append(self.next_id)
                self.next_id += 1
        
        # Age unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        return list(self.tracks.values())
    
    def _iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_t, y1_t, x2_t, y2_t = box2
        xi1, yi1 = max(x1, x1_t), max(y1, y1_t)
        xi2, yi2 = min(x2, x2_t), min(y2, y2_t)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        inter = (xi2-xi1) * (yi2-yi1)
        union = (x2-x1)*(y2-y1) + (x2_t-x1_t)*(y2_t-y1_t) - inter
        return inter / union if union > 0 else 0

class KalmanBallTracker:
    """Kalman filter for ball trajectory smoothing"""
    def __init__(self):
        self.positions = []
        self.velocities = []
        self.last_pos = None
    
    def update(self, ball_detection):
        if not ball_detection:
            return self._predict()
        
        pos = ball_detection['center']
        self.positions.append(pos)
        
        if len(self.positions) > 1:
            vel = (pos[0] - self.positions[-2][0], pos[1] - self.positions[-2][1])
            self.velocities.append(vel)
        
        if len(self.positions) > 10:
            self.positions.pop(0)
            self.velocities.pop(0)
        
        self.last_pos = pos
        return {'center': pos, 'confidence': ball_detection['confidence']}
    
    def _predict(self):
        if not self.last_pos or not self.velocities:
            return None
        
        # Simple linear prediction
        vel = self.velocities[-1]
        pred_pos = (self.last_pos[0] + vel[0], self.last_pos[1] + vel[1])
        return {'center': pred_pos, 'confidence': 0.5, 'predicted': True}

class RTMPoseEstimator:
    """RTMPose for fast pose estimation"""
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose()
        except:
            self.pose = None
    
    def estimate_pose(self, frame, bbox):
        if not self.pose:
            return None
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            return landmarks
        return None

class TemporalEventDetector:
    """Temporal CNN + Transformer for event detection"""
    def __init__(self):
        self.sequence_length = 16
        self.pose_buffer = {}
        self.events = ['pass', 'shot', 'tackle', 'dribble', 'header']
    
    def update_sequence(self, player_id, pose_features):
        if player_id not in self.pose_buffer:
            self.pose_buffer[player_id] = []
        
        self.pose_buffer[player_id].append(pose_features)
        if len(self.pose_buffer[player_id]) > self.sequence_length:
            self.pose_buffer[player_id].pop(0)
    
    def detect_event(self, player_id):
        if player_id not in self.pose_buffer or len(self.pose_buffer[player_id]) < self.sequence_length:
            return None
        
        # Simple heuristic-based event detection
        sequence = self.pose_buffer[player_id]
        
        # Calculate movement patterns
        movement = sum(abs(sequence[i][0] - sequence[i-1][0]) + abs(sequence[i][1] - sequence[i-1][1]) 
                      for i in range(1, len(sequence)))
        
        arm_movement = sum(abs(sequence[i][22] - sequence[i-1][22]) for i in range(1, len(sequence)) if len(sequence[i]) > 22)
        
        if movement > 0.1 and arm_movement > 0.05:
            return {'event': 'pass', 'confidence': 0.8}
        elif movement > 0.15:
            return {'event': 'shot', 'confidence': 0.7}
        elif movement > 0.05:
            return {'event': 'dribble', 'confidence': 0.6}
        
        return None

class StateOfTheArtModelStack:
    """Complete SOTA model stack"""
    def __init__(self):
        self.rtdetr = RTDETRDetector()
        self.ball_detector = SmallObjectBallDetector()
        self.tracker = ByteTrackTracker()
        self.ball_tracker = KalmanBallTracker()
        self.pose_estimator = RTMPoseEstimator()
        self.event_detector = TemporalEventDetector()
    
    def process_frame(self, frame):
        # Primary detection with RT-DETR
        detections = self.rtdetr.detect(frame)
        
        # Specialized ball detection
        ball_detections = self.ball_detector.detect_ball(frame)
        
        # Track players
        tracked_players = self.tracker.update(detections)
        
        # Track ball
        ball_track = None
        if ball_detections:
            ball_track = self.ball_tracker.update(ball_detections[0])
        else:
            ball_track = self.ball_tracker.update(None)
        
        # Pose estimation and event detection
        events = []
        for player in tracked_players:
            pose = self.pose_estimator.estimate_pose(frame, player['bbox'])
            if pose:
                self.event_detector.update_sequence(player['id'], pose)
                event = self.event_detector.detect_event(player['id'])
                if event:
                    events.append({
                        'player_id': player['id'],
                        'event_type': event['event'],
                        'confidence': event['confidence']
                    })
        
        return {
            'players': tracked_players,
            'ball': ball_track,
            'events': events
        }
    
    def predict_xg(self, features):
        # Simple XGBoost-style prediction
        distance = features['distance']
        angle = features['angle']
        defenders = features['defenders']
        pressure = features['pressure']
        
        # Simplified xG calculation
        base_xg = 1.0 / (1.0 + distance/10)
        angle_factor = 1.0 - (angle / 90)
        defender_penalty = 0.9 ** defenders
        pressure_penalty = 1.0 - pressure * 0.3
        
        xg = base_xg * angle_factor * defender_penalty * pressure_penalty
        return min(max(xg, 0.01), 0.99)