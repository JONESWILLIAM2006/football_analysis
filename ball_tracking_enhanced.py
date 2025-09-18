# Enhanced Ball Detection and Tracking System
# Integrates with existing football_analysis.py

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

class SimpleBallTracker:
    """Simplified ball tracker with trajectory visualization"""
    def __init__(self, max_trajectory=20):
        self.trajectory = deque(maxlen=max_trajectory)
        self.last_position = None
        self.confidence_threshold = 0.3
        
    def update(self, detection):
        """Update ball position with new detection"""
        if detection and len(detection) >= 3:
            x, y, conf = detection
            if conf > self.confidence_threshold:
                self.trajectory.append((int(x), int(y)))
                self.last_position = (int(x), int(y))
                return (int(x), int(y))
        return self.last_position
    
    def get_trajectory(self):
        return list(self.trajectory)
    
    def draw_trajectory(self, frame):
        """Draw ball trajectory on frame"""
        if len(self.trajectory) > 1:
            points = list(self.trajectory)
            for i in range(1, len(points)):
                # Create fading trail effect
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                cv2.line(frame, points[i-1], points[i], 
                        (color_intensity, color_intensity, 255), 2)
        return frame

class EnhancedFootballTracker:
    """Enhanced tracking system with ball detection and team classification"""
    
    def __init__(self, model_path="yolov8x.pt"):
        self.yolo = YOLO(model_path)
        self.ball_tracker = SimpleBallTracker(max_trajectory=20)
        self.player_tracks = {}
        self.next_id = 1
        
        # Team colors for visualization
        self.colors = {
            'Team A': (0, 0, 255),     # Red
            'Team B': (255, 0, 0),     # Blue
            'Referee': (0, 255, 255),  # Yellow
            'Ball': (255, 255, 255)    # White
        }
        
    def process_frame(self, frame):
        """Process frame with enhanced ball and player tracking"""
        results = self.yolo(frame, verbose=False)
        
        players = []
        ball_detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                if score > 0.5:
                    x1, y1, x2, y2 = box
                    
                    if cls == 0:  # Person class
                        # Simple team assignment based on x position
                        team = 'Team A' if (x1 + x2) / 2 < frame.shape[1] / 2 else 'Team B'
                        
                        players.append({
                            'bbox': [x1, y1, x2, y2],
                            'team': team,
                            'confidence': score
                        })
                    
                    elif cls == 37:  # Ball class (sports ball in COCO)
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        ball_detections.append((cx, cy, score))
        
        # Update ball tracker
        ball_pos = None
        if ball_detections:
            best_ball = max(ball_detections, key=lambda x: x[2])
            ball_pos = self.ball_tracker.update(best_ball)
        
        ball = None
        if ball_pos:
            ball = {
                'center': ball_pos,
                'bbox': [ball_pos[0]-10, ball_pos[1]-10, ball_pos[0]+10, ball_pos[1]+10],
                'confidence': max([b[2] for b in ball_detections]) if ball_detections else 0.5
            }
        
        return players, ball
    
    def draw_annotations(self, frame, players, ball):
        """Draw enhanced annotations with ball trail"""
        # Draw players with team colors
        for i, player in enumerate(players):
            x1, y1, x2, y2 = [int(v) for v in player['bbox']]
            team = player['team']
            color = self.colors.get(team, (128, 128, 128))
            
            # Enhanced player visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Player ID and team label with background
            label = f"P{i+1} {team}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-25), (x1+label_size[0]+10, y1-5), color, -1)
            cv2.putText(frame, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw ball with enhanced trail
        if ball:
            # Draw ball trajectory trail
            frame = self.ball_tracker.draw_trajectory(frame)
            
            # Draw current ball position
            if 'center' in ball:
                cx, cy = map(int, ball['center'])
                
                # Ball with glow effect
                cv2.circle(frame, (cx, cy), 12, (100, 100, 100), -1)  # Shadow
                cv2.circle(frame, (cx, cy), 8, self.colors['Ball'], -1)  # Main ball
                cv2.circle(frame, (cx, cy), 8, (0, 0, 0), 2)  # Black outline
                
                # Ball label with background
                label = "BALL"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (cx-label_size[0]//2-5, cy-25), 
                             (cx+label_size[0]//2+5, cy-10), (0, 0, 0), -1)
                cv2.putText(frame, label, (cx-label_size[0]//2, cy-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['Ball'], 2)
        
        return frame

def process_video_with_ball_tracking(video_path, output_path=None):
    """Process video with enhanced ball tracking"""
    tracker = EnhancedFootballTracker()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None
    
    # Video writer setup
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    ball_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        players, ball = tracker.process_frame(frame)
        
        # Record ball position
        if ball:
            ball_detections.append({
                'frame': frame_count,
                'position': ball['center'],
                'confidence': ball['confidence']
            })
        
        # Draw annotations
        annotated_frame = tracker.draw_annotations(frame, players, ball)
        
        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if ball:
            cv2.putText(annotated_frame, f"Ball Detections: {len(ball_detections)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        if output_path:
            out.write(annotated_frame)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    if output_path:
        out.release()
    
    print(f"Processing complete! Total frames: {frame_count}")
    print(f"Ball detections: {len(ball_detections)}")
    
    return {
        'total_frames': frame_count,
        'ball_detections': ball_detections,
        'detection_rate': len(ball_detections) / frame_count * 100 if frame_count > 0 else 0
    }

# Integration function for existing football_analysis.py
def integrate_ball_tracking_with_existing_system():
    """
    Integration guide for existing football_analysis.py:
    
    1. Replace the existing FootballTrackingPipeline with EnhancedFootballTracker
    2. Update the DetectionEngine.run_detection method to use the new tracker
    3. Add ball trail visualization to the output video
    
    Example integration:
    
    # In DetectionEngine.__init__():
    from ball_tracking_enhanced import EnhancedFootballTracker
    self.enhanced_tracker = EnhancedFootballTracker()
    
    # In DetectionEngine.run_detection():
    # Replace existing tracking with:
    tracked_players, ball_detection = self.enhanced_tracker.process_frame(frame)
    frame = self.enhanced_tracker.draw_annotations(frame, tracked_players, ball_detection)
    """
    pass

if __name__ == "__main__":
    # Example usage
    video_path = "sample_match.mp4"  # Replace with your video path
    output_path = "enhanced_tracking_output.mp4"
    
    results = process_video_with_ball_tracking(video_path, output_path)
    
    if results:
        print(f"Detection rate: {results['detection_rate']:.1f}%")
        print(f"Total ball detections: {len(results['ball_detections'])}")