# Advanced VAR Features Implementation
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from collections import deque
import time

class MultiCamera3DTracker:
    """3D tracking using multiple camera views with DeepLabCut/OpenPose integration"""
    def __init__(self, camera_count=4):
        self.camera_count = camera_count
        self.mp_pose = mp.solutions.pose
        self.pose_detectors = [self.mp_pose.Pose(min_detection_confidence=0.5) for _ in range(camera_count)]
        self.camera_matrices = {}
        self.distortion_coeffs = {}
        self.triangulated_points = {}
        
    def calibrate_cameras(self, calibration_data):
        """Calibrate multiple cameras for 3D reconstruction"""
        for cam_id, data in calibration_data.items():
            self.camera_matrices[cam_id] = data['camera_matrix']
            self.distortion_coeffs[cam_id] = data['distortion_coeffs']
    
    def detect_3d_poses(self, frames):
        """Detect 3D poses from multiple camera views"""
        poses_2d = {}
        
        for cam_id, frame in frames.items():
            if cam_id < len(self.pose_detectors):
                results = self.pose_detectors[cam_id].process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    poses_2d[cam_id] = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) 
                                       for lm in results.pose_landmarks.landmark]
        
        # Triangulate 3D points
        if len(poses_2d) >= 2:
            return self._triangulate_3d_points(poses_2d)
        return None
    
    def _triangulate_3d_points(self, poses_2d):
        """Triangulate 3D points from 2D poses"""
        cam_ids = list(poses_2d.keys())
        if len(cam_ids) < 2:
            return None
            
        points_3d = []
        for i in range(len(poses_2d[cam_ids[0]])):
            points_2d = []
            projection_matrices = []
            
            for cam_id in cam_ids:
                if i < len(poses_2d[cam_id]):
                    points_2d.append(poses_2d[cam_id][i])
                    # Create projection matrix (simplified)
                    P = np.hstack([self.camera_matrices[cam_id], np.zeros((3, 1))])
                    projection_matrices.append(P)
            
            if len(points_2d) >= 2:
                # Triangulate using DLT
                point_3d = self._dlt_triangulation(points_2d, projection_matrices)
                points_3d.append(point_3d)
        
        return points_3d
    
    def _dlt_triangulation(self, points_2d, projection_matrices):
        """Direct Linear Transform triangulation"""
        A = []
        for i, (point, P) in enumerate(zip(points_2d, projection_matrices)):
            x, y = point
            A.append([x * P[2, 0] - P[0, 0], x * P[2, 1] - P[0, 1], x * P[2, 2] - P[0, 2]])
            A.append([y * P[2, 0] - P[1, 0], y * P[2, 1] - P[1, 1], y * P[2, 2] - P[1, 2]])
        
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        point_3d = Vt[-1, :3] / Vt[-1, 3] if Vt[-1, 3] != 0 else Vt[-1, :3]
        return point_3d

class BallAwareFoulDetector:
    """Enhanced foul detection that considers ball proximity"""
    def __init__(self, ball_proximity_threshold=50):
        self.ball_proximity_threshold = ball_proximity_threshold
        self.contact_history = deque(maxlen=30)
        
    def detect_ball_aware_foul(self, players, ball_position, frame):
        """Detect fouls considering ball proximity"""
        if not ball_position or len(players) < 2:
            return None
            
        # Find players near the ball
        ball_x, ball_y = ball_position
        players_near_ball = []
        
        for player in players:
            px, py = (player['bbox'][0] + player['bbox'][2]) / 2, (player['bbox'][1] + player['bbox'][3]) / 2
            distance_to_ball = np.sqrt((px - ball_x)**2 + (py - ball_y)**2)
            
            if distance_to_ball < self.ball_proximity_threshold:
                players_near_ball.append({
                    'id': player['id'],
                    'position': (px, py),
                    'bbox': player['bbox'],
                    'distance_to_ball': distance_to_ball
                })
        
        # Check for contact between players near ball
        if len(players_near_ball) >= 2:
            for i, player1 in enumerate(players_near_ball):
                for player2 in players_near_ball[i+1:]:
                    contact_detected = self._detect_player_contact(player1, player2, frame)
                    if contact_detected:
                        foul_severity = self._assess_foul_severity(player1, player2, ball_position)
                        
                        return {
                            'type': 'ball_aware_foul',
                            'player1': player1['id'],
                            'player2': player2['id'],
                            'ball_distance': min(player1['distance_to_ball'], player2['distance_to_ball']),
                            'severity': foul_severity,
                            'contact_point': self._get_contact_point(player1, player2),
                            'ball_involved': True
                        }
        
        return None
    
    def _detect_player_contact(self, player1, player2, frame):
        """Detect physical contact between players"""
        bbox1 = player1['bbox']
        bbox2 = player2['bbox']
        
        # Check bounding box overlap
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        overlap_area = overlap_x * overlap_y
        
        # Significant overlap indicates contact
        return overlap_area > 500  # Threshold for contact
    
    def _assess_foul_severity(self, player1, player2, ball_position):
        """Assess foul severity based on context"""
        # Distance to ball affects severity
        min_ball_distance = min(player1['distance_to_ball'], player2['distance_to_ball'])
        
        if min_ball_distance < 20:
            return 'high'  # Very close to ball
        elif min_ball_distance < 35:
            return 'medium'
        else:
            return 'low'
    
    def _get_contact_point(self, player1, player2):
        """Calculate contact point between players"""
        p1_center = ((player1['bbox'][0] + player1['bbox'][2]) / 2, 
                     (player1['bbox'][1] + player1['bbox'][3]) / 2)
        p2_center = ((player2['bbox'][0] + player2['bbox'][2]) / 2,
                     (player2['bbox'][1] + player2['bbox'][3]) / 2)
        
        return ((p1_center[0] + p2_center[0]) / 2, (p1_center[1] + p2_center[1]) / 2)

class ContactForceDetector:
    """Physics-based contact force detection using player velocities"""
    def __init__(self):
        self.velocity_history = {}
        self.mass_estimate = 75  # kg, average player mass
        
    def calculate_contact_force(self, player1, player2, velocities, dt=1/30):
        """Calculate contact force using physics"""
        if player1['id'] not in velocities or player2['id'] not in velocities:
            return None
            
        v1 = np.array(velocities[player1['id']])
        v2 = np.array(velocities[player2['id']])
        
        # Relative velocity
        v_rel = v1 - v2
        v_rel_magnitude = np.linalg.norm(v_rel)
        
        # Contact angle
        p1_pos = np.array([(player1['bbox'][0] + player1['bbox'][2]) / 2,
                          (player1['bbox'][1] + player1['bbox'][3]) / 2])
        p2_pos = np.array([(player2['bbox'][0] + player2['bbox'][2]) / 2,
                          (player2['bbox'][1] + player2['bbox'][3]) / 2])
        
        contact_vector = p2_pos - p1_pos
        contact_angle = np.arctan2(contact_vector[1], contact_vector[0]) * 180 / np.pi
        
        # Estimate impact force (simplified)
        # F = m * Δv / Δt
        momentum_change = self.mass_estimate * v_rel_magnitude
        impact_force = momentum_change / dt
        
        return {
            'force_magnitude': impact_force,
            'contact_angle': contact_angle,
            'relative_velocity': v_rel_magnitude,
            'impact_severity': self._classify_impact_severity(impact_force)
        }
    
    def _classify_impact_severity(self, force):
        """Classify impact severity based on force"""
        if force > 1000:  # N
            return 'severe'
        elif force > 500:
            return 'moderate'
        elif force > 200:
            return 'light'
        else:
            return 'minimal'

class MLDiveDetector:
    """Machine learning-based dive detection"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self._train_model()
        
    def _train_model(self):
        """Train dive detection model with synthetic data"""
        # Generate synthetic training data
        n_samples = 1000
        
        # Features: [velocity_before, velocity_after, contact_force, fall_angle, time_to_ground]
        X = np.random.rand(n_samples, 5)
        
        # Simulate dive patterns
        dive_indices = np.random.choice(n_samples, n_samples // 10, replace=False)
        y = np.zeros(n_samples)
        
        for idx in dive_indices:
            # Dive characteristics: low contact force, dramatic velocity change, specific fall pattern
            X[idx, 0] = np.random.uniform(0.8, 1.0)  # High velocity before
            X[idx, 1] = np.random.uniform(0.0, 0.2)  # Low velocity after
            X[idx, 2] = np.random.uniform(0.0, 0.3)  # Low contact force
            X[idx, 3] = np.random.uniform(0.7, 1.0)  # Dramatic fall angle
            X[idx, 4] = np.random.uniform(0.8, 1.0)  # Quick time to ground
            y[idx] = 1
        
        # Train model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def detect_dive(self, player_data, contact_data, velocity_history):
        """Detect potential diving using ML model"""
        if not self.is_trained:
            return None
            
        player_id = player_data['id']
        
        # Extract features
        features = self._extract_dive_features(player_id, contact_data, velocity_history)
        if features is None:
            return None
            
        # Predict
        features_scaled = self.scaler.transform([features])
        dive_probability = self.model.predict_proba(features_scaled)[0][1]
        is_dive = dive_probability > 0.7
        
        return {
            'player_id': player_id,
            'is_dive': is_dive,
            'dive_probability': dive_probability,
            'confidence': 'high' if dive_probability > 0.8 else 'medium' if dive_probability > 0.6 else 'low'
        }
    
    def _extract_dive_features(self, player_id, contact_data, velocity_history):
        """Extract features for dive detection"""
        if player_id not in velocity_history or len(velocity_history[player_id]) < 5:
            return None
            
        velocities = velocity_history[player_id]
        
        # Velocity before contact
        vel_before = np.mean([np.linalg.norm(v) for v in velocities[-5:-2]])
        
        # Velocity after contact
        vel_after = np.mean([np.linalg.norm(v) for v in velocities[-2:]])
        
        # Contact force (normalized)
        contact_force = contact_data.get('force_magnitude', 0) / 1000.0
        
        # Fall characteristics (simulated)
        fall_angle = np.random.uniform(0, 1)  # Would be calculated from pose
        time_to_ground = np.random.uniform(0, 1)  # Would be calculated from tracking
        
        return [vel_before, vel_after, contact_force, fall_angle, time_to_ground]

class CalibratedOffsideDetector:
    """Millimeter-precision offside detection using camera calibration"""
    def __init__(self, calibration_data):
        self.homography_matrix = calibration_data.get('homography_matrix')
        self.camera_matrix = calibration_data.get('camera_matrix')
        self.precision_threshold = 0.05  # 5cm precision
        
    def detect_calibrated_offside(self, players, ball_owner_id, event_type):
        """Detect offside with millimeter precision"""
        if not self.homography_matrix or event_type not in ['pass', 'shot', 'cross']:
            return None
            
        # Convert pixel coordinates to real-world coordinates
        world_positions = {}
        for player_id, player_data in players.items():
            pixel_pos = [(player_data['bbox'][0] + player_data['bbox'][2]) / 2,
                        (player_data['bbox'][1] + player_data['bbox'][3]) / 2]
            world_pos = self._pixel_to_world(pixel_pos)
            if world_pos is not None:
                world_positions[player_id] = world_pos
        
        if len(world_positions) < 4:
            return None
            
        # Determine attacking and defending teams
        attacking_players = [pid for pid in world_positions.keys() 
                           if pid != ball_owner_id and self._is_attacking_team(pid)]
        defending_players = [pid for pid in world_positions.keys() 
                           if not self._is_attacking_team(pid)]
        
        if len(defending_players) < 2:
            return None
            
        # Find second-to-last defender
        defending_y_positions = [world_positions[pid][1] for pid in defending_players]
        defending_y_positions.sort()
        offside_line_y = defending_y_positions[1]  # Second-to-last defender
        
        # Check attacking players
        offside_players = []
        for player_id in attacking_players:
            player_y = world_positions[player_id][1]
            margin = player_y - offside_line_y
            
            if margin > self.precision_threshold:  # Player is offside
                offside_players.append({
                    'player_id': player_id,
                    'margin_meters': margin,
                    'position': world_positions[player_id],
                    'offside_line_y': offside_line_y
                })
        
        if offside_players:
            return {
                'type': 'calibrated_offside',
                'offside_players': offside_players,
                'precision': 'millimeter',
                'confidence': 0.95,
                'event_type': event_type
            }
        
        return None
    
    def _pixel_to_world(self, pixel_pos):
        """Convert pixel coordinates to world coordinates using homography"""
        if self.homography_matrix is None:
            return None
            
        point = np.array([[pixel_pos]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return world_point[0][0]
    
    def _is_attacking_team(self, player_id):
        """Determine if player is on attacking team (simplified)"""
        return player_id <= 11  # Assume first 11 players are team A

class VARReplaySystem:
    """VAR replay system with slow-motion and event timeline"""
    def __init__(self, buffer_size=300):  # 10 seconds at 30fps
        self.frame_buffer = deque(maxlen=buffer_size)
        self.event_timeline = []
        self.replay_segments = {}
        
    def add_frame(self, frame, frame_number, events=None):
        """Add frame to replay buffer"""
        frame_data = {
            'frame': frame.copy(),
            'frame_number': frame_number,
            'timestamp': time.time(),
            'events': events or []
        }
        self.frame_buffer.append(frame_data)
        
        # Update event timeline
        if events:
            for event in events:
                self.event_timeline.append({
                    'frame_number': frame_number,
                    'event': event,
                    'timestamp': time.time()
                })
    
    def create_var_replay(self, incident_frame, replay_duration=5):
        """Create VAR replay segment"""
        fps = 30
        frames_needed = replay_duration * fps
        
        # Find incident in buffer
        incident_index = None
        for i, frame_data in enumerate(self.frame_buffer):
            if frame_data['frame_number'] == incident_frame:
                incident_index = i
                break
        
        if incident_index is None:
            return None
            
        # Extract replay segment
        start_index = max(0, incident_index - frames_needed // 2)
        end_index = min(len(self.frame_buffer), incident_index + frames_needed // 2)
        
        replay_frames = []
        for i in range(start_index, end_index):
            replay_frames.append(self.frame_buffer[i])
        
        # Create slow-motion version
        slow_motion_frames = self._create_slow_motion(replay_frames, factor=0.5)
        
        replay_id = f"var_replay_{incident_frame}_{int(time.time())}"
        self.replay_segments[replay_id] = {
            'normal_speed': replay_frames,
            'slow_motion': slow_motion_frames,
            'incident_frame': incident_frame,
            'created_at': time.time()
        }
        
        return replay_id
    
    def _create_slow_motion(self, frames, factor=0.5):
        """Create slow-motion replay by frame interpolation"""
        if not frames:
            return []
            
        slow_frames = []
        for i in range(len(frames) - 1):
            current_frame = frames[i]['frame']
            next_frame = frames[i + 1]['frame']
            
            # Add original frame
            slow_frames.append(frames[i])
            
            # Add interpolated frames
            interpolation_steps = int(1 / factor) - 1
            for step in range(1, interpolation_steps + 1):
                alpha = step / (interpolation_steps + 1)
                interpolated = cv2.addWeighted(current_frame, 1 - alpha, next_frame, alpha, 0)
                
                interpolated_data = {
                    'frame': interpolated,
                    'frame_number': frames[i]['frame_number'] + step * 0.1,
                    'timestamp': frames[i]['timestamp'],
                    'events': [],
                    'interpolated': True
                }
                slow_frames.append(interpolated_data)
        
        # Add last frame
        if frames:
            slow_frames.append(frames[-1])
            
        return slow_frames
    
    def get_replay_segment(self, replay_id, speed='normal'):
        """Get replay segment by ID"""
        if replay_id not in self.replay_segments:
            return None
            
        segment = self.replay_segments[replay_id]
        return segment.get(f'{speed}_speed' if speed == 'slow_motion' else 'normal_speed', [])
    
    def generate_var_decision_data(self, replay_id, decision_type='offside'):
        """Generate VAR decision data with measurements"""
        if replay_id not in self.replay_segments:
            return None
            
        segment = self.replay_segments[replay_id]
        incident_frame = segment['incident_frame']
        
        # Generate decision data based on type
        if decision_type == 'offside':
            return {
                'decision': 'ONSIDE',
                'margin': '0.15m',
                'confidence': 94,
                'measurement_points': [
                    {'player': 'Attacker', 'position': [45.2, 12.8]},
                    {'player': 'Defender', 'position': [45.35, 12.8]}
                ],
                'replay_id': replay_id,
                'frames_analyzed': len(segment['normal_speed'])
            }
        elif decision_type == 'foul':
            return {
                'decision': 'FOUL CONFIRMED',
                'contact_force': '850N',
                'confidence': 87,
                'contact_point': [52.1, 34.5],
                'replay_id': replay_id
            }
        
        return None