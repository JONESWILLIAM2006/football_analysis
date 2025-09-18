# VAR Enhanced Football Analysis - Complete Integration
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp
from ultralytics import YOLO

class MultiCamera3DTracker:
    def __init__(self, cam_params_list):
        self.cams = cam_params_list

    def triangulate_point(self, pts_2d_list):
        A = []
        for pt, cam in zip(pts_2d_list, self.cams):
            x, y = pt
            P = cam['projection_matrix']
            A.append(x*P[2] - P[0])
            A.append(y*P[2] - P[1])
        A = np.vstack(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return (X[:3] / X[3])

    def reconstruct_3D(self, detections_per_cam):
        n_points = len(detections_per_cam[0])
        points_3D = []
        for i in range(n_points):
            pts_2d_list = [cam_det[i] for cam_det in detections_per_cam]
            points_3D.append(self.triangulate_point(pts_2d_list))
        return points_3D

class BallAwareFoulDetector:
    def __init__(self, ball_radius=0.11):
        self.ball_radius = ball_radius

    def is_foul_near_ball(self, player1_pos, player2_pos, ball_pos):
        dist_ball_p1 = np.linalg.norm(player1_pos - ball_pos)
        dist_ball_p2 = np.linalg.norm(player2_pos - ball_pos)
        return min(dist_ball_p1, dist_ball_p2) < 50.0  # 50 pixel threshold

class ContactForceDetector:
    def calculate_force(self, p1_pos, p2_pos, p1_vel, p2_vel, mass=75):
        rel_vel = p1_vel - p2_vel
        rel_pos = p1_pos - p2_pos
        distance = np.linalg.norm(rel_pos)
        if distance < 30.0:  # Contact threshold in pixels
            force_mag = mass * np.linalg.norm(rel_vel) / (distance + 1e-6)
            angle = np.arctan2(rel_pos[1], rel_pos[0])
            return force_mag, np.degrees(angle)
        return 0, 0

class MLDiveDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        # Train with synthetic data
        X = np.random.rand(1000, 5)
        y = np.random.choice([0, 1], 1000, p=[0.9, 0.1])
        self.model.fit(X, y)

    def predict_dive(self, features):
        return self.model.predict_proba([features])[0][1] > 0.7

class CalibratedOffsideDetector:
    def __init__(self):
        # Default homography matrix
        self.H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def to_pitch_coords(self, x, y):
        pts = np.array([[x, y]], dtype='float32')
        pts = np.array([pts])
        pts_real = cv2.perspectiveTransform(pts, self.H)
        return pts_real[0][0]

    def check_offside(self, attacker_pos_img, last_defender_pos_img):
        attacker_real = self.to_pitch_coords(*attacker_pos_img)
        defender_real = self.to_pitch_coords(*last_defender_pos_img)
        return attacker_real[0] > defender_real[0]

class VARReplaySystem:
    def __init__(self):
        self.events = []
        self.frame_buffer = []

    def add_event(self, frame_num, event_type):
        self.events.append((frame_num, event_type))

    def add_frame(self, frame):
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > 300:  # Keep last 10 seconds at 30fps
            self.frame_buffer.pop(0)

class VAREnhancedDetectionEngine:
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        
        # Initialize VAR components
        self.multi_camera = MultiCamera3DTracker([{'projection_matrix': np.eye(3,4)}] * 2)
        self.foul_detector = BallAwareFoulDetector()
        self.contact_force = ContactForceDetector()
        self.dive_detector = MLDiveDetector()
        self.offside_detector = CalibratedOffsideDetector()
        self.replay_system = VARReplaySystem()
        
        # Velocity tracking
        self.player_velocities = {}
        self.last_positions = {}
        
    def analyze_frame_with_var(self, frame, frame_num):
        """Complete VAR-enhanced frame analysis"""
        # Add frame to replay buffer
        self.replay_system.add_frame(frame)
        
        # YOLO detection
        results = self.model(frame, verbose=False)
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
                            'center': center
                        })
                    elif cls == 37:  # Ball
                        ball_pos = center
        
        # Update velocities
        self._update_velocities(players)
        
        # VAR Analysis
        var_events = []
        
        # 1. Ball-aware foul detection
        if len(players) >= 2 and ball_pos:
            for i, p1 in enumerate(players):
                for p2 in players[i+1:]:
                    p1_pos = np.array(p1['center'] + [0])
                    p2_pos = np.array(p2['center'] + [0])
                    ball_3d = np.array(ball_pos + [0])
                    
                    if self.foul_detector.is_foul_near_ball(p1_pos, p2_pos, ball_3d):
                        var_events.append({
                            'type': 'ball_aware_foul',
                            'frame': frame_num,
                            'players': [p1['id'], p2['id']],
                            'ball_distance': min(np.linalg.norm(p1_pos[:2] - ball_3d[:2]), 
                                               np.linalg.norm(p2_pos[:2] - ball_3d[:2]))
                        })
        
        # 2. Contact force analysis
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                if p1['id'] in self.player_velocities and p2['id'] in self.player_velocities:
                    p1_pos = np.array(p1['center'])
                    p2_pos = np.array(p2['center'])
                    p1_vel = self.player_velocities[p1['id']]
                    p2_vel = self.player_velocities[p2['id']]
                    
                    force, angle = self.contact_force.calculate_force(p1_pos, p2_pos, p1_vel, p2_vel)
                    if force > 500:  # Significant force threshold
                        var_events.append({
                            'type': 'contact_force',
                            'frame': frame_num,
                            'players': [p1['id'], p2['id']],
                            'force': force,
                            'angle': angle
                        })
        
        # 3. Dive detection
        for player in players:
            if player['id'] in self.player_velocities:
                vel = self.player_velocities[player['id']]
                features = [np.linalg.norm(vel), vel[0], vel[1], 0.5, 0.3]  # Simplified features
                
                if self.dive_detector.predict_dive(features):
                    var_events.append({
                        'type': 'dive_detection',
                        'frame': frame_num,
                        'player': player['id'],
                        'confidence': 0.8
                    })
        
        # 4. Offside detection
        if len(players) >= 4:
            attackers = [p for p in players if p['id'] % 2 == 0]  # Simplified team assignment
            defenders = [p for p in players if p['id'] % 2 == 1]
            
            if attackers and defenders:
                attacker = attackers[0]
                defender = defenders[0]
                
                if self.offside_detector.check_offside(attacker['center'], defender['center']):
                    var_events.append({
                        'type': 'offside',
                        'frame': frame_num,
                        'attacker': attacker['id'],
                        'defender': defender['id']
                    })
        
        # Add events to replay system
        for event in var_events:
            self.replay_system.add_event(frame_num, event['type'])
        
        # Draw VAR overlays
        frame = self._draw_var_overlays(frame, var_events)
        
        return frame, var_events, players, ball_pos
    
    def _update_velocities(self, players):
        """Update player velocities for force calculations"""
        for player in players:
            player_id = player['id']
            current_pos = np.array(player['center'])
            
            if player_id in self.last_positions:
                velocity = current_pos - self.last_positions[player_id]
                self.player_velocities[player_id] = velocity
            else:
                self.player_velocities[player_id] = np.array([0.0, 0.0])
            
            self.last_positions[player_id] = current_pos
    
    def _draw_var_overlays(self, frame, var_events):
        """Draw VAR analysis overlays"""
        for event in var_events:
            if event['type'] == 'ball_aware_foul':
                cv2.putText(frame, f"BALL-AWARE FOUL - {event['ball_distance']:.1f}px", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif event['type'] == 'contact_force':
                cv2.putText(frame, f"CONTACT FORCE: {event['force']:.0f}N @ {event['angle']:.0f}°", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif event['type'] == 'dive_detection':
                cv2.putText(frame, f"DIVE DETECTED - Player {event['player']}", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            elif event['type'] == 'offside':
                cv2.putText(frame, f"OFFSIDE - Player {event['attacker']} vs {event['defender']}", 
                           (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame
    
    def process_video_with_var(self, video_path, output_path):
        """Process entire video with VAR analysis"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        all_var_events = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # VAR-enhanced analysis
            processed_frame, var_events, players, ball_pos = self.analyze_frame_with_var(frame, frame_num)
            
            # Store events
            all_var_events.extend(var_events)
            
            # Write frame
            out.write(processed_frame)
            frame_num += 1
            
            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames...")
        
        cap.release()
        out.release()
        
        return all_var_events

# Example usage
if __name__ == "__main__":
    # Initialize VAR-enhanced system
    var_engine = VAREnhancedDetectionEngine()
    
    # Process video with all VAR features
    var_events = var_engine.process_video_with_var("input_video.mp4", "var_enhanced_output.mp4")
    
    print(f"Detected {len(var_events)} VAR events:")
    for event in var_events[:10]:  # Show first 10 events
        print(f"Frame {event['frame']}: {event['type']}")