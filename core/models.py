# Core detection and tracking models
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

class RTDETRv3Detector:
    """RT-DETR-v3 for real-time detection"""
    def __init__(self, model_path="rtdetr-l.pt", conf_threshold=0.5):
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
    """Enhanced ball detection with SAHI and motion enhancement"""
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        self.prev_frame = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        
    def detect_with_motion_enhancement(self, frame):
        # Motion enhancement
        enhanced_frame = self._enhance_motion(frame)
        
        # High-resolution detection
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
            
            # Size and aspect ratio checks
            if 5 <= width <= 50 and 5 <= height <= 50:
                aspect_ratio = width / height
                if 0.5 <= aspect_ratio <= 2.0:
                    filtered.append(det)
        
        return filtered

class ByteTracker:
    """Simplified ByteTracker implementation"""
    def __init__(self, frame_rate=30):
        self.frame_rate = frame_rate
        self.tracks = {}
        self.next_id = 1
        self.max_age = 30
        
    def update(self, detections):
        # Simple tracking based on IoU
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