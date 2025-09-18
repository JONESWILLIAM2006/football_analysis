# Enhanced Football Analysis with Continuous Commentary
import cv2
import numpy as np
import time
import threading
from collections import deque

class ContinuousCommentaryEngine:
    def __init__(self, fps=30):
        self.fps = fps
        self.match_time = 0
        self.events_history = deque(maxlen=1000)
        self.commentary_stream = []
        self.last_commentary = 0
        self.running = False
        self.thread = None
        
    def start_match_commentary(self):
        self.running = True
        self.match_time = 0
        self.thread = threading.Thread(target=self._commentary_loop, daemon=True)
        self.thread.start()
    
    def _commentary_loop(self):
        while self.running and self.match_time < 5400:  # 90 minutes
            current_minute = int(self.match_time / 60)
            
            # Continuous commentary every 10 seconds
            if self.match_time - self.last_commentary >= 10:
                commentary = f"Minute {current_minute}: Match continues at good pace"
                self.commentary_stream.append({
                    'time': self.match_time,
                    'type': 'continuous',
                    'text': commentary
                })
                self.last_commentary = self.match_time
            
            # Halftime summary
            if current_minute == 45 and self.match_time % 60 < 1:
                goals = len([e for e in self.events_history if e.get('type') == 'goal'])
                summary = f"Halftime: {goals} goals scored so far"
                self.commentary_stream.append({
                    'time': self.match_time,
                    'type': 'halftime',
                    'text': summary
                })
            
            # Fulltime summary
            if current_minute >= 90:
                total_events = len(self.events_history)
                summary = f"Full time! Match ends with {total_events} key events"
                self.commentary_stream.append({
                    'time': self.match_time,
                    'type': 'fulltime',
                    'text': summary
                })
                break
            
            # Predictive commentary every 15 seconds
            if self.match_time % 15 == 0:
                player_id = np.random.randint(1, 23)
                target_id = np.random.randint(1, 23)
                predictive = f"Player {player_id} should look for Player {target_id}"
                self.commentary_stream.append({
                    'time': self.match_time,
                    'type': 'predictive',
                    'text': predictive
                })
            
            time.sleep(1)
            self.match_time += 1
    
    def add_event(self, event):
        self.events_history.append(event)
        
        # Generate immediate commentary for significant events
        if event.get('type') in ['goal', 'shot', 'foul']:
            player = event.get('player_from', 'Player')
            if event.get('type') == 'goal':
                commentary = f"GOAL! Brilliant finish from Player {player}!"
            elif event.get('type') == 'shot':
                commentary = f"Shot from Player {player}! Close attempt!"
            else:
                commentary = f"Foul called on Player {player}"
            
            self.commentary_stream.append({
                'time': self.match_time,
                'type': 'immediate',
                'text': commentary
            })
    
    def get_recent_commentary(self, last_n=5):
        return self.commentary_stream[-last_n:] if self.commentary_stream else []
    
    def get_match_time_display(self):
        minutes = int(self.match_time / 60)
        seconds = int(self.match_time % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def stop_commentary(self):
        self.running = False
        if self.thread:
            self.thread.join()

class EnhancedDetectionEngine:
    def __init__(self, model_path="yolov8x.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.commentary_engine = ContinuousCommentaryEngine()
        self.match_started = False
        
    def run_detection_with_commentary(self, video_path, predictive_tactician, tactical_analyzer, xg_xa_model, selected_formation):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], [], [], [], []

        # Start continuous commentary
        if not self.match_started:
            self.commentary_engine.start_match_commentary()
            self.match_started = True

        width, height = 1280, 720
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = "outputs/processed_video_with_commentary.mp4"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        events = []
        wrong_passes = []
        foul_events = []
        training_rows = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (width, height))
            results = self.model(frame, verbose=False)
            
            # Basic detection and tracking
            players = []
            ball_pos = None
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    if score > 0.5:
                        x1, y1, x2, y2 = box
                        center = [(x1+x2)/2, (y1+y2)/2]
                        
                        if cls == 0:  # Person
                            players.append({
                                'id': i,
                                'bbox': [x1, y1, x2, y2],
                                'center': center,
                                'team': 'A' if i <= 11 else 'B'
                            })
                        elif cls == 37:  # Ball
                            ball_pos = center
            
            # Generate events for commentary
            if frame_num % 30 == 0 and players:  # Every second
                event_type = np.random.choice(['pass', 'shot', 'foul'], p=[0.7, 0.2, 0.1])
                player_id = np.random.choice([p['id'] for p in players])
                
                event = {
                    'type': event_type,
                    'player_from': player_id,
                    'frame': frame_num,
                    'team': 'A' if player_id <= 11 else 'B'
                }
                
                events.append((frame_num, event_type, player_id, 'N/A', 'N/A', event))
                self.commentary_engine.add_event(event)
            
            # Get commentary feed
            commentary_feed = {
                'match_time': self.commentary_engine.get_match_time_display(),
                'recent_commentary': self.commentary_engine.get_recent_commentary()
            }
            
            # Draw commentary on frame
            if commentary_feed['recent_commentary']:
                y_pos = 50
                cv2.putText(frame, f"Time: {commentary_feed['match_time']}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                for i, comment in enumerate(commentary_feed['recent_commentary'][-3:]):
                    y_pos += 30
                    text = comment['text'][:60] + "..." if len(comment['text']) > 60 else comment['text']
                    color = (0, 255, 255) if comment['type'] == 'predictive' else (255, 255, 255)
                    cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw players
            for player in players:
                x1, y1, x2, y2 = [int(v) for v in player['bbox']]
                color = (0, 255, 0) if player['team'] == 'A' else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"P{player['id']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ball
            if ball_pos:
                cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 8, (255, 255, 255), -1)
                cv2.putText(frame, "BALL", (int(ball_pos[0])-20, int(ball_pos[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            out.write(frame)
            frame_num += 1
            
            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames with commentary...")

        cap.release()
        out.release()
        
        # Stop commentary
        self.commentary_engine.stop_commentary()
        
        return events, wrong_passes, foul_events, training_rows, {'commentary_enabled': True}

# Integration function for existing system
def enhance_detection_engine_with_commentary(detection_engine):
    """Enhance existing DetectionEngine with continuous commentary"""
    detection_engine.commentary_engine = ContinuousCommentaryEngine()
    detection_engine.match_started = False
    
    # Override run_detection method
    original_run_detection = detection_engine.run_detection
    
    def run_detection_with_commentary(video_path, predictive_tactician, tactical_analyzer, xg_xa_model, selected_formation):
        # Start commentary
        if not detection_engine.match_started:
            detection_engine.commentary_engine.start_match_commentary()
            detection_engine.match_started = True
        
        # Call original method
        events, wrong_passes, foul_events, training_rows, match_stats = original_run_detection(
            video_path, predictive_tactician, tactical_analyzer, xg_xa_model, selected_formation
        )
        
        # Add commentary events
        for event in events:
            detection_engine.commentary_engine.add_event({
                'type': event[1],
                'player_from': event[2],
                'frame': event[0]
            })
        
        # Stop commentary
        detection_engine.commentary_engine.stop_commentary()
        
        return events, wrong_passes, foul_events, training_rows, match_stats
    
    detection_engine.run_detection = run_detection_with_commentary
    return detection_engine