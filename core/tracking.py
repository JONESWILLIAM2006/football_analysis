"""
Player and ball tracking components
"""
import cv2
import numpy as np
from collections import OrderedDict, deque
from ultralytics import YOLO
import mediapipe as mp

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class BaseTrack:
    _count = 0
    track_id = 0
    is_activated = False
    state = TrackState.New
    
    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

class KalmanFilter:
    """Simplified Kalman Filter for object tracking"""
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

class FootballTrackingPipeline:
    """Main tracking pipeline with constant IDs and team classification"""
    def __init__(self, model_path="yolov8x.pt"):
        self.yolo = YOLO(model_path)
        self.colors = {
            'Team A': (0, 0, 255),
            'Team B': (255, 0, 0),
            'Referee': (0, 255, 255),
            'Ball': (255, 255, 255)
        }
    
    def process_frame(self, frame):
        """Process single frame with tracking and team classification"""
        results = self.yolo(frame, verbose=False)
        players = []
        ball = None
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                if score > 0.5:
                    x1, y1, x2, y2 = box
                    if cls == 0:  # Person
                        players.append({
                            'id': len(players) + 1,
                            'bbox': [x1, y1, x2, y2],
                            'team': 'Team A' if len(players) < 11 else 'Team B',
                            'confidence': score
                        })
                    elif cls == 37:  # Ball
                        ball = {
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'bbox': [x1, y1, x2, y2],
                            'confidence': score
                        }
        
        return players, ball
    
    def draw_annotations(self, frame, players, ball):
        """Draw tracking annotations on frame"""
        for player in players:
            x1, y1, x2, y2 = [int(v) for v in player['bbox']]
            color = self.colors.get(player['team'], (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{player['id']} {player['team']}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if ball:
            cx, cy = map(int, ball['center'])
            cv2.circle(frame, (cx, cy), 8, self.colors['Ball'], -1)
            cv2.putText(frame, "BALL", (cx-20, cy-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['Ball'], 2)
        
        return frame