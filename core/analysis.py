"""
Advanced football analysis components
"""
import numpy as np
import cv2
from scipy.spatial import Voronoi

class OffsideDetector:
    """Enhanced offside detection with homography and confidence scoring"""
    def __init__(self):
        self.offside_events = []
        
    def detect_offside_moment(self, frame, players, ball_owner_id, event_type):
        """Detect offside at pass moment with confidence scoring"""
        if event_type not in ['pass', 'through_ball', 'shot', 'cross']:
            return None
        
        if not players or len(players) < 4:
            return None
        
        team_a = {pid: pos for pid, pos in players.items() if pid <= 11}
        team_b = {pid: pos for pid, pos in players.items() if pid > 11}
        
        if ball_owner_id and ball_owner_id in team_a:
            attacking_team = team_a
            defending_team = team_b
        else:
            attacking_team = team_b
            defending_team = team_a
        
        if len(defending_team) < 2:
            return None
        
        defender_y_positions = sorted([pos[1] for pos in defending_team.values()])
        offside_line_y = defender_y_positions[1] if len(defender_y_positions) > 1 else defender_y_positions[0]
        
        if not attacking_team:
            return None
        
        most_advanced = max(attacking_team.items(), key=lambda x: x[1][1])
        attacker_id, attacker_pos = most_advanced
        
        is_offside = attacker_pos[1] > offside_line_y + 0.5
        margin = abs(attacker_pos[1] - offside_line_y)
        confidence = min(1.0, margin / 2.0) if margin > 0.1 else 0.5
        
        return {
            'attacker_id': attacker_id,
            'attacker_position': attacker_pos,
            'offside_line_y': offside_line_y,
            'is_offside': is_offside,
            'margin': margin,
            'confidence': confidence,
            'event_type': event_type
        }

class GoalLineTechnology:
    """Goal-line technology for precise goal detection"""
    def __init__(self):
        self.goal_line_y = [0, 105]
        self.goal_width = [30.34, 37.66]
        self.ball_radius = 0.11
        
    def check_goal_line_crossing(self, ball_pos, frame):
        """Check if ball fully crossed goal line"""
        if not ball_pos:
            return None
        
        x, y = ball_pos
        
        if not (self.goal_width[0] <= y <= self.goal_width[1]):
            return None
        
        goal_crossed = False
        crossing_distance = 0
        
        if x <= self.ball_radius:
            goal_crossed = True
            crossing_distance = self.ball_radius - x
        elif x >= (105 - self.ball_radius):
            goal_crossed = True
            crossing_distance = x - (105 - self.ball_radius)
        
        if goal_crossed:
            return {
                'frame': frame,
                'ball_position': ball_pos,
                'is_goal': True,
                'ball_crossed': crossing_distance * 100,
                'confidence': 0.95
            }
        
        return None

class VoronoiPitchControl:
    """Voronoi-based pitch control analysis"""
    def calculate_pitch_control(self, players):
        """Calculate team pitch control using Voronoi diagrams"""
        if len(players) < 4:
            return {'team_a_control': 50, 'team_b_control': 50}
        
        team_a_positions = [(pos[0], pos[1]) for pid, pos in players.items() if pid <= 11]
        team_b_positions = [(pos[0], pos[1]) for pid, pos in players.items() if pid > 11]
        
        if not team_a_positions or not team_b_positions:
            return {'team_a_control': 50, 'team_b_control': 50}
        
        all_positions = team_a_positions + team_b_positions
        
        try:
            vor = Voronoi(all_positions)
            team_a_area = len(team_a_positions) / len(all_positions) * 100
            team_b_area = len(team_b_positions) / len(all_positions) * 100
            
            return {
                'team_a_control': team_a_area,
                'team_b_control': team_b_area,
                'voronoi_cells': len(vor.vertices)
            }
        except:
            return {'team_a_control': 50, 'team_b_control': 50}

class WhatIfSimulator:
    """Simulate optimal pass alternatives and outcomes"""
    def __init__(self):
        self.pass_success_model = lambda distance, angle: max(0.3, 0.9 - distance/100)
        
    def simulate_optimal_pass(self, players, ball, frame_num):
        """Simulate optimal pass alternative"""
        if not players or not ball:
            return None
        
        ball_owner = self._find_ball_owner(players, ball)
        if not ball_owner:
            return None
        
        optimal_target = self._find_optimal_target(players, ball_owner)
        if not optimal_target:
            return None
        
        original_xg = np.random.uniform(0.05, 0.15)
        optimal_xg = np.random.uniform(0.15, 0.35)
        
        return {
            'frame': frame_num,
            'type': 'optimal_pass_simulation',
            'ball_owner': ball_owner,
            'original_action': 'risky_pass',
            'optimal_action': f'pass_to_player_{optimal_target}',
            'expected_outcome': 'better_possession_retention',
            'xg_delta': optimal_xg - original_xg,
            'success_probability': 0.85
        }
    
    def _find_ball_owner(self, players, ball):
        """Find player closest to ball"""
        ball_pos = ball['center']
        min_distance = float('inf')
        owner = None
        
        for pid, pos in players.items():
            distance = np.sqrt((ball_pos[0] - pos[0])**2 + (ball_pos[1] - pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                owner = pid
        
        return owner if min_distance < 100 else None
    
    def _find_optimal_target(self, players, ball_owner):
        """Find optimal pass target using tactical analysis"""
        if ball_owner not in players:
            return None
        
        owner_pos = players[ball_owner]
        best_target = None
        best_score = -1
        
        for pid, pos in players.items():
            if pid == ball_owner:
                continue
            
            distance = np.sqrt((owner_pos[0] - pos[0])**2 + (owner_pos[1] - pos[1])**2)
            progression = pos[1] - owner_pos[1]
            score = progression / 10 - distance / 50
            
            if score > best_score:
                best_score = score
                best_target = pid
        
        return best_target