# VAR Integration Module
import cv2
import numpy as np
from advanced_var_features import (
    MultiCamera3DTracker, BallAwareFoulDetector, ContactForceDetector,
    MLDiveDetector, CalibratedOffsideDetector, VARReplaySystem
)

class VARAnalysisEngine:
    """Enhanced VAR analysis engine with all advanced features"""
    def __init__(self):
        # Initialize advanced VAR components
        self.multi_camera_tracker = MultiCamera3DTracker(camera_count=4)
        self.ball_aware_foul_detector = BallAwareFoulDetector(ball_proximity_threshold=50)
        self.contact_force_detector = ContactForceDetector()
        self.ml_dive_detector = MLDiveDetector()
        self.var_replay_system = VARReplaySystem(buffer_size=300)
        
        # Calibration data (would be loaded from file in production)
        self.calibration_data = {
            'homography_matrix': np.eye(3),
            'camera_matrix': np.eye(3),
            'camera_0': {'camera_matrix': np.eye(3), 'distortion_coeffs': np.zeros(5)},
            'camera_1': {'camera_matrix': np.eye(3), 'distortion_coeffs': np.zeros(5)},
            'camera_2': {'camera_matrix': np.eye(3), 'distortion_coeffs': np.zeros(5)},
            'camera_3': {'camera_matrix': np.eye(3), 'distortion_coeffs': np.zeros(5)}
        }
        
        self.calibrated_offside_detector = CalibratedOffsideDetector(self.calibration_data)
        
        # Velocity tracking for force calculations
        self.velocity_history = {}
        self.player_positions_history = {}
        
    def analyze_frame_advanced(self, frame, players, ball_position, frame_number):
        """Comprehensive VAR analysis for a single frame"""
        var_events = []
        
        # Add frame to replay buffer
        self.var_replay_system.add_frame(frame, frame_number)
        
        # Update velocity tracking
        self._update_velocity_tracking(players)
        
        # 1. Ball-aware foul detection
        ball_aware_foul = self.ball_aware_foul_detector.detect_ball_aware_foul(
            players, ball_position, frame
        )
        if ball_aware_foul:
            var_events.append({
                'type': 'ball_aware_foul',
                'frame': frame_number,
                'data': ball_aware_foul
            })
        
        # 2. Contact force analysis
        if len(players) >= 2:
            for i, player1 in enumerate(players):
                for player2 in players[i+1:]:
                    if self._players_in_contact(player1, player2):
                        force_data = self.contact_force_detector.calculate_contact_force(
                            player1, player2, self.velocity_history
                        )
                        if force_data and force_data['force_magnitude'] > 200:
                            var_events.append({
                                'type': 'contact_force',
                                'frame': frame_number,
                                'players': [player1['id'], player2['id']],
                                'data': force_data
                            })
        
        # 3. ML-based dive detection
        for player in players:
            if player['id'] in self.velocity_history:
                contact_data = {'force_magnitude': 100}  # Simplified
                dive_result = self.ml_dive_detector.detect_dive(
                    player, contact_data, self.velocity_history
                )
                if dive_result and dive_result['is_dive']:
                    var_events.append({
                        'type': 'dive_detection',
                        'frame': frame_number,
                        'player': player['id'],
                        'data': dive_result
                    })
        
        # 4. Calibrated offside detection
        if ball_position and len(players) >= 4:
            ball_owner_id = self._find_ball_owner(players, ball_position)
            if ball_owner_id:
                offside_result = self.calibrated_offside_detector.detect_calibrated_offside(
                    {p['id']: p for p in players}, ball_owner_id, 'pass'
                )
                if offside_result:
                    var_events.append({
                        'type': 'calibrated_offside',
                        'frame': frame_number,
                        'data': offside_result
                    })
        
        return var_events
    
    def analyze_multi_camera_3d(self, camera_frames):
        """3D analysis using multiple camera views"""
        if len(camera_frames) < 2:
            return None
            
        # Detect 3D poses
        poses_3d = self.multi_camera_tracker.detect_3d_poses(camera_frames)
        
        if poses_3d:
            return {
                'type': '3d_tracking',
                'poses_3d': poses_3d,
                'camera_count': len(camera_frames),
                'precision': 'millimeter'
            }
        
        return None
    
    def create_var_incident_replay(self, incident_frame, incident_type='offside'):
        """Create VAR replay for incident review"""
        replay_id = self.var_replay_system.create_var_replay(incident_frame, replay_duration=10)
        
        if replay_id:
            # Generate decision data
            decision_data = self.var_replay_system.generate_var_decision_data(
                replay_id, incident_type
            )
            
            return {
                'replay_id': replay_id,
                'decision_data': decision_data,
                'incident_frame': incident_frame,
                'incident_type': incident_type
            }
        
        return None
    
    def get_var_replay_frames(self, replay_id, speed='normal'):
        """Get VAR replay frames for display"""
        return self.var_replay_system.get_replay_segment(replay_id, speed)
    
    def _update_velocity_tracking(self, players):
        """Update velocity tracking for force calculations"""
        for player in players:
            player_id = player['id']
            current_pos = np.array([(player['bbox'][0] + player['bbox'][2]) / 2,
                                   (player['bbox'][1] + player['bbox'][3]) / 2])
            
            if player_id not in self.player_positions_history:
                self.player_positions_history[player_id] = []
                self.velocity_history[player_id] = []
            
            # Add current position
            self.player_positions_history[player_id].append(current_pos)
            
            # Calculate velocity
            if len(self.player_positions_history[player_id]) >= 2:
                prev_pos = self.player_positions_history[player_id][-2]
                velocity = current_pos - prev_pos
                self.velocity_history[player_id].append(velocity)
                
                # Keep only last 10 frames
                if len(self.velocity_history[player_id]) > 10:
                    self.velocity_history[player_id].pop(0)
                if len(self.player_positions_history[player_id]) > 10:
                    self.player_positions_history[player_id].pop(0)
    
    def _players_in_contact(self, player1, player2):
        """Check if two players are in contact"""
        bbox1 = player1['bbox']
        bbox2 = player2['bbox']
        
        # Check bounding box overlap
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        overlap_area = overlap_x * overlap_y
        
        return overlap_area > 300  # Contact threshold
    
    def _find_ball_owner(self, players, ball_position):
        """Find player closest to ball"""
        if not ball_position:
            return None
            
        ball_x, ball_y = ball_position
        min_distance = float('inf')
        closest_player = None
        
        for player in players:
            px, py = (player['bbox'][0] + player['bbox'][2]) / 2, (player['bbox'][1] + player['bbox'][3]) / 2
            distance = np.sqrt((px - ball_x)**2 + (py - ball_y)**2)
            
            if distance < min_distance and distance < 40:
                min_distance = distance
                closest_player = player['id']
        
        return closest_player
    
    def draw_var_overlays(self, frame, var_events):
        """Draw VAR analysis overlays on frame"""
        for event in var_events:
            if event['type'] == 'ball_aware_foul':
                self._draw_foul_overlay(frame, event['data'])
            elif event['type'] == 'contact_force':
                self._draw_force_overlay(frame, event['data'], event['players'])
            elif event['type'] == 'dive_detection':
                self._draw_dive_overlay(frame, event['data'], event['player'])
            elif event['type'] == 'calibrated_offside':
                self._draw_offside_overlay(frame, event['data'])
        
        return frame
    
    def _draw_foul_overlay(self, frame, foul_data):
        """Draw foul detection overlay"""
        cv2.putText(frame, f"FOUL DETECTED - {foul_data['severity'].upper()}", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        if 'contact_point' in foul_data:
            cx, cy = map(int, foul_data['contact_point'])
            cv2.circle(frame, (cx, cy), 15, (0, 0, 255), 3)
    
    def _draw_force_overlay(self, frame, force_data, players):
        """Draw contact force overlay"""
        force_text = f"CONTACT FORCE: {force_data['force_magnitude']:.0f}N"
        severity_color = {
            'severe': (0, 0, 255),
            'moderate': (0, 165, 255),
            'light': (0, 255, 255),
            'minimal': (0, 255, 0)
        }.get(force_data['impact_severity'], (255, 255, 255))
        
        cv2.putText(frame, force_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, severity_color, 2)
    
    def _draw_dive_overlay(self, frame, dive_data, player_id):
        """Draw dive detection overlay"""
        if dive_data['is_dive']:
            confidence_text = f"DIVE DETECTED - Player {player_id} ({dive_data['dive_probability']:.2f})"
            cv2.putText(frame, confidence_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    def _draw_offside_overlay(self, frame, offside_data):
        """Draw calibrated offside overlay"""
        if offside_data['offside_players']:
            for player_data in offside_data['offside_players']:
                margin = player_data['margin_meters']
                status = "OFFSIDE" if margin > 0 else "ONSIDE"
                color = (0, 0, 255) if margin > 0 else (0, 255, 0)
                
                text = f"{status}: {abs(margin):.3f}m (Player {player_data['player_id']})"
                cv2.putText(frame, text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)