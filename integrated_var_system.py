# Integrated Advanced VAR System
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp

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
        return min(dist_ball_p1, dist_ball_p2) < 5.0

class ContactForceDetector:
    def calculate_force(self, p1_pos, p2_pos, p1_vel, p2_vel, mass=75):
        rel_vel = p1_vel - p2_vel
        rel_pos = p1_pos - p2_pos
        distance = np.linalg.norm(rel_pos)
        if distance < 1.0:
            force_mag = mass * np.linalg.norm(rel_vel) / distance
            angle = np.arctan2(rel_pos[1], rel_pos[0])
            return force_mag, np.degrees(angle)
        return 0, 0

class MLDiveDetector:
    def __init__(self, model_path=None):
        self.model = RandomForestClassifier()
        if model_path:
            import joblib
            self.model = joblib.load(model_path)

    def predict_dive(self, features):
        return self.model.predict([features])[0]

class CalibratedOffsideDetector:
    def __init__(self, pitch_points_img=None, pitch_points_real=None):
        if pitch_points_img and pitch_points_real:
            self.H, _ = cv2.findHomography(np.array(pitch_points_img), np.array(pitch_points_real))
        else:
            self.H = np.eye(3)

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
    def __init__(self, video_path=None):
        self.events = []
        self.video_path = video_path

    def add_event(self, frame_num, event_type):
        self.events.append((frame_num, event_type))

    def export_replay(self, output_path, event_window=30):
        if not self.video_path:
            return
        cap = cv2.VideoCapture(self.video_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, 
                              (int(cap.get(3)), int(cap.get(4))))
        for frame_num, event_type in self.events:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_num-event_window,0))
            for i in range(event_window*2):
                ret, frame = cap.read()
                if not ret: break
                cv2.putText(frame, event_type, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                out.write(frame)
        out.release()
        cap.release()

class IntegratedVARSystem:
    def __init__(self):
        # Initialize all VAR components
        self.multi_camera = MultiCamera3DTracker([{'projection_matrix': np.eye(3,4)}] * 2)
        self.foul_detector = BallAwareFoulDetector()
        self.contact_force = ContactForceDetector()
        self.dive_detector = MLDiveDetector()
        self.offside_detector = CalibratedOffsideDetector()
        self.replay_system = VARReplaySystem()
        
    def analyze_frame(self, frame, players, ball_pos, frame_num):
        """Complete VAR analysis for single frame"""
        var_events = []
        
        # Ball-aware foul detection
        if len(players) >= 2 and ball_pos:
            for i, p1 in enumerate(players):
                for p2 in players[i+1:]:
                    p1_pos = np.array([(p1['bbox'][0] + p1['bbox'][2])/2, (p1['bbox'][1] + p1['bbox'][3])/2, 0])
                    p2_pos = np.array([(p2['bbox'][0] + p2['bbox'][2])/2, (p2['bbox'][1] + p2['bbox'][3])/2, 0])
                    ball_3d = np.array([ball_pos[0], ball_pos[1], 0])
                    
                    if self.foul_detector.is_foul_near_ball(p1_pos, p2_pos, ball_3d):
                        var_events.append({
                            'type': 'ball_aware_foul',
                            'frame': frame_num,
                            'players': [p1['id'], p2['id']],
                            'ball_distance': min(np.linalg.norm(p1_pos - ball_3d), np.linalg.norm(p2_pos - ball_3d))
                        })
        
        # Contact force analysis
        if len(players) >= 2:
            for i, p1 in enumerate(players):
                for p2 in players[i+1:]:
                    p1_pos = np.array([(p1['bbox'][0] + p1['bbox'][2])/2, (p1['bbox'][1] + p1['bbox'][3])/2])
                    p2_pos = np.array([(p2['bbox'][0] + p2['bbox'][2])/2, (p2['bbox'][1] + p2['bbox'][3])/2])
                    p1_vel = np.array([1, 0])  # Simplified velocity
                    p2_vel = np.array([-1, 0])
                    
                    force, angle = self.contact_force.calculate_force(p1_pos, p2_pos, p1_vel, p2_vel)
                    if force > 100:
                        var_events.append({
                            'type': 'contact_force',
                            'frame': frame_num,
                            'players': [p1['id'], p2['id']],
                            'force': force,
                            'angle': angle
                        })
        
        # Offside detection
        if len(players) >= 4 and ball_pos:
            attackers = [p for p in players if p['id'] <= 11]
            defenders = [p for p in players if p['id'] > 11]
            
            if attackers and defenders:
                attacker = attackers[0]
                defender = defenders[0]
                att_pos = ((attacker['bbox'][0] + attacker['bbox'][2])/2, (attacker['bbox'][1] + attacker['bbox'][3])/2)
                def_pos = ((defender['bbox'][0] + defender['bbox'][2])/2, (defender['bbox'][1] + defender['bbox'][3])/2)
                
                if self.offside_detector.check_offside(att_pos, def_pos):
                    var_events.append({
                        'type': 'offside',
                        'frame': frame_num,
                        'attacker': attacker['id'],
                        'defender': defender['id']
                    })
        
        # Add events to replay system
        for event in var_events:
            self.replay_system.add_event(frame_num, event['type'])
        
        return var_events
    
    def draw_var_overlays(self, frame, var_events):
        """Draw VAR analysis overlays"""
        for event in var_events:
            if event['type'] == 'ball_aware_foul':
                cv2.putText(frame, f"FOUL NEAR BALL - {event['ball_distance']:.1f}m", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif event['type'] == 'contact_force':
                cv2.putText(frame, f"CONTACT FORCE: {event['force']:.0f}N", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif event['type'] == 'offside':
                cv2.putText(frame, f"OFFSIDE - Player {event['attacker']}", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame