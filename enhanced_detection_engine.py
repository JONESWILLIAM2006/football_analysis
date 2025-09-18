# Enhanced Detection Engine with Advanced VAR Features
import cv2
import numpy as np
from ultralytics import YOLO
import time

class EnhancedDetectionEngine:
    """Detection engine with integrated advanced VAR features"""
    def __init__(self, model_path="yolov8x.pt"):
        self.model = YOLO(model_path)
        
        # Initialize advanced VAR features
        try:
            from var_integration import VARAnalysisEngine
            self.var_engine = VARAnalysisEngine()
            self.var_enabled = True
        except ImportError:
            self.var_engine = None
            self.var_enabled = False
        
        # Multi-camera setup
        self.camera_feeds = {}
        self.var_incidents = []
        
    def process_multi_camera_frame(self, camera_frames, frame_number):
        """Process multiple camera feeds for 3D tracking"""
        if not self.var_enabled or len(camera_frames) < 2:
            return None
            
        # 3D pose detection
        tracking_3d = self.var_engine.analyze_multi_camera_3d(camera_frames)
        
        if tracking_3d:
            return {
                'frame': frame_number,
                'type': '3d_tracking',
                'data': tracking_3d
            }
        
        return None
    
    def detect_ball_aware_fouls(self, frame, players, ball_position, frame_number):
        """Enhanced foul detection considering ball proximity"""
        if not self.var_enabled:
            return []
            
        var_events = self.var_engine.analyze_frame_advanced(
            frame, players, ball_position, frame_number
        )
        
        # Filter for ball-aware fouls
        ball_aware_fouls = [
            event for event in var_events 
            if event['type'] == 'ball_aware_foul'
        ]
        
        return ball_aware_fouls
    
    def calculate_contact_forces(self, frame, players, frame_number):
        """Calculate physics-based contact forces"""
        if not self.var_enabled:
            return []
            
        var_events = self.var_engine.analyze_frame_advanced(
            frame, players, None, frame_number
        )
        
        # Filter for contact force events
        force_events = [
            event for event in var_events 
            if event['type'] == 'contact_force'
        ]
        
        return force_events
    
    def detect_dives_ml(self, frame, players, frame_number):
        """ML-based dive detection"""
        if not self.var_enabled:
            return []
            
        var_events = self.var_engine.analyze_frame_advanced(
            frame, players, None, frame_number
        )
        
        # Filter for dive detection events
        dive_events = [
            event for event in var_events 
            if event['type'] == 'dive_detection'
        ]
        
        return dive_events
    
    def detect_calibrated_offside(self, frame, players, ball_position, frame_number):
        """Millimeter-precision offside detection"""
        if not self.var_enabled:
            return []
            
        var_events = self.var_engine.analyze_frame_advanced(
            frame, players, ball_position, frame_number
        )
        
        # Filter for offside events
        offside_events = [
            event for event in var_events 
            if event['type'] == 'calibrated_offside'
        ]
        
        return offside_events
    
    def create_var_replay(self, incident_frame, incident_type='offside'):
        """Create VAR replay for incident review"""
        if not self.var_enabled:
            return None
            
        replay_data = self.var_engine.create_var_incident_replay(
            incident_frame, incident_type
        )
        
        if replay_data:
            self.var_incidents.append(replay_data)
            return replay_data
        
        return None
    
    def get_var_replay_frames(self, replay_id, speed='normal'):
        """Get VAR replay frames"""
        if not self.var_enabled:
            return []
            
        return self.var_engine.get_var_replay_frames(replay_id, speed)
    
    def draw_var_overlays(self, frame, var_events):
        """Draw all VAR analysis overlays"""
        if not self.var_enabled or not var_events:
            return frame
            
        return self.var_engine.draw_var_overlays(frame, var_events)
    
    def comprehensive_var_analysis(self, frame, players, ball_position, frame_number):
        """Run complete VAR analysis pipeline"""
        if not self.var_enabled:
            return frame, []
            
        # Run all VAR analyses
        var_events = self.var_engine.analyze_frame_advanced(
            frame, players, ball_position, frame_number
        )
        
        # Draw overlays
        annotated_frame = self.var_engine.draw_var_overlays(frame, var_events)
        
        # Check for significant incidents requiring VAR review
        significant_events = [
            event for event in var_events 
            if event['type'] in ['calibrated_offside', 'ball_aware_foul', 'dive_detection']
            and self._is_significant_incident(event)
        ]
        
        # Create replays for significant incidents
        for event in significant_events:
            replay_data = self.create_var_replay(frame_number, event['type'])
            if replay_data:
                event['replay_id'] = replay_data['replay_id']
        
        return annotated_frame, var_events
    
    def _is_significant_incident(self, event):
        """Determine if incident requires VAR review"""
        if event['type'] == 'calibrated_offside':
            return True  # All offside checks are significant
        elif event['type'] == 'ball_aware_foul':
            return event['data'].get('severity') in ['high', 'medium']
        elif event['type'] == 'dive_detection':
            return event['data'].get('dive_probability', 0) > 0.8
        elif event['type'] == 'contact_force':
            return event['data'].get('force_magnitude', 0) > 500
        
        return False
    
    def get_var_decision_summary(self, replay_id):
        """Get VAR decision summary for display"""
        if not self.var_enabled:
            return None
            
        # Find incident by replay_id
        incident = next((inc for inc in self.var_incidents if inc['replay_id'] == replay_id), None)
        
        if incident and 'decision_data' in incident:
            return {
                'incident_type': incident['incident_type'],
                'decision': incident['decision_data']['decision'],
                'confidence': incident['decision_data']['confidence'],
                'measurements': incident['decision_data'].get('measurement_points', []),
                'replay_available': True
            }
        
        return None
    
    def export_var_data(self, output_path):
        """Export VAR analysis data"""
        if not self.var_enabled:
            return False
            
        var_data = {
            'incidents': self.var_incidents,
            'total_incidents': len(self.var_incidents),
            'incident_types': list(set(inc['incident_type'] for inc in self.var_incidents)),
            'export_timestamp': time.time()
        }
        
        try:
            import json
            with open(output_path, 'w') as f:
                json.dump(var_data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Failed to export VAR data: {e}")
            return False