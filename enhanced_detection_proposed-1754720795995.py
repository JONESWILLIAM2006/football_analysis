# Enhanced Detection & Tracking System
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import torch

class ByteTracker:
    def __init__(self, frame_rate=30, track_thresh=0.5, match_thresh=0.8):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.tracked_stracks = []
        self.frame_id = 0
        
    def update(self, detections):
        self.frame_id += 1
        if detections is None or len(detections) == 0:
            return []
        
        # Simple tracking based on IoU
        if not self.tracked_stracks:
            # Initialize tracks
            for i, det in enumerate(detections):
                track = STrack(det, self.frame_id, i+1)
                self.tracked_stracks.append(track)
        else:
            # Match detections to existing tracks
            cost_matrix = self._calculate_cost_matrix(detections)
            matches = self._linear_assignment(cost_matrix)
            
            # Update matched tracks
            for track_idx, det_idx in matches:
                self.tracked_stracks[track_idx].update(detections[det_idx], self.frame_id)
        
        return [track for track in self.tracked_stracks if track.is_active()]
    
    def _calculate_cost_matrix(self, detections):
        cost_matrix = np.zeros((len(self.tracked_stracks), len(detections)))
        for i, track in enumerate(self.tracked_stracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - self._bbox_iou(track.bbox, det[:4])
        return cost_matrix
    
    def _linear_assignment(self, cost_matrix):
        matches = []
        for i in range(cost_matrix.shape[0]):
            j = np.argmin(cost_matrix[i])
            if cost_matrix[i, j] < self.match_thresh:
                matches.append((i, j))
        return matches
    
    def _bbox_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
        inter_x2, inter_y2 = min(x2, x4), min(y2, y4)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        
        return inter_area / (box1_area + box2_area - inter_area)

class STrack:
    def __init__(self, detection, frame_id, track_id):
        self.track_id = track_id
        self.bbox = detection[:4]
        self.score = detection[4]
        self.last_frame = frame_id
        self.history = deque(maxlen=30)
        self.history.append(self.bbox)
    
    def update(self, detection, frame_id):
        self.bbox = detection[:4]
        self.score = detection[4]
        self.last_frame = frame_id
        self.history.append(self.bbox)
    
    def is_active(self):
        return True  # Simplified

class EnhancedBallDetector:
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        self.ball_class_id = 37  # Sports ball in COCO
        self.kalman = cv2.KalmanFilter(4, 2)
        self._init_kalman()
        self.ball_history = deque(maxlen=10)
        
    def _init_kalman(self):
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
    
    def detect_ball(self, frame):
        results = self.model(frame, verbose=False)
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            ball_detections = []
            for box, score, cls in zip(boxes, scores, classes):
                if int(cls) == self.ball_class_id and score > 0.3:
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    ball_detections.append((cx, cy, score))
            
            if ball_detections:
                # Take highest confidence detection
                best_ball = max(ball_detections, key=lambda x: x[2])
                self._update_kalman(best_ball[:2])
                return best_ball
        
        # Use Kalman prediction if no detection
        if len(self.ball_history) > 0:
            prediction = self.kalman.predict()
            return (prediction[0], prediction[1], 0.5)
        
        return None
    
    def _update_kalman(self, position):
        measurement = np.array([[position[0]], [position[1]]], dtype=np.float32)
        self.kalman.correct(measurement)
        self.ball_history.append(position)

class SegmentationDetector:
    def __init__(self, model_path="yolov8x-seg.pt"):
        self.model = YOLO(model_path)
    
    def detect_with_masks(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                for i, (box, score, cls, mask) in enumerate(zip(boxes, scores, classes, masks)):
                    if int(cls) == 0 and score > 0.5:  # Person class
                        detections.append({
                            'bbox': box,
                            'score': score,
                            'class': int(cls),
                            'mask': mask
                        })
        
        return detections

class MultiObjectTracker:
    def __init__(self):
        self.player_tracker = ByteTracker(track_thresh=0.6)
        self.referee_tracker = ByteTracker(track_thresh=0.5)
        self.ball_detector = EnhancedBallDetector()
        self.seg_detector = SegmentationDetector()
    
    def update(self, frame):
        # Get segmentation detections
        seg_detections = self.seg_detector.detect_with_masks(frame)
        
        # Separate players and referees (simplified)
        player_dets = []
        referee_dets = []
        
        for det in seg_detections:
            # Simple heuristic: referees are usually in different colored clothing
            if self._is_referee(frame, det):
                referee_dets.append([*det['bbox'], det['score']])
            else:
                player_dets.append([*det['bbox'], det['score']])
        
        # Update trackers
        player_tracks = self.player_tracker.update(player_dets)
        referee_tracks = self.referee_tracker.update(referee_dets)
        ball_detection = self.ball_detector.detect_ball(frame)
        
        return {
            'players': player_tracks,
            'referees': referee_tracks,
            'ball': ball_detection
        }
    
    def _is_referee(self, frame, detection):
        # Simplified referee detection based on clothing color
        x1, y1, x2, y2 = detection['bbox']
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if crop.size == 0:
            return False
        
        # Check for black clothing (common referee color)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        black_ratio = np.sum(black_mask > 0) / black_mask.size
        
        return black_ratio > 0.3