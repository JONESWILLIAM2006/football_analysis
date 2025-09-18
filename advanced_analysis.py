import cv2
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from scipy.spatial.distance import cdist
import math

class AdvancedFootballAnalysis:
    def __init__(self):
        self.homography_matrix = None
        self.field_coords = {'width': 68, 'length': 105}
        self.goal_coords = {'left': (0, 30.34, 37.66), 'right': (105, 30.34, 37.66)}
        
    def detect_offside(self, frame, players, ball_pos, ball_velocity):
        """Detect offside violations"""
        if not self._is_pass_moment(ball_velocity):
            return None
            
        attacking_players = [p for p in players if p['team'] == 'attacking']
        defending_players = [p for p in players if p['team'] == 'defending']
        
        if not defending_players:
            return None
            
        # Find last defender line
        defender_x_positions = [p['position'][0] for p in defending_players]
        last_defender_line = max(defender_x_positions) if attacking_players[0]['position'][0] < 52.5 else min(defender_x_positions)
        
        offside_players = []
        for player in attacking_players:
            if self._is_in_offside_position(player['position'], last_defender_line, ball_pos):
                offside_players.append(player)
        
        return {
            'offside_detected': len(offside_players) > 0,
            'offside_players': offside_players,
            'defender_line': last_defender_line,
            'confidence': 0.85 if offside_players else 1.0
        }
    
    def _is_pass_moment(self, ball_velocity):
        """Detect pass moment based on ball velocity change"""
        if len(ball_velocity) < 2:
            return False
        return abs(ball_velocity[-1] - ball_velocity[-2]) > 5.0
    
    def _is_in_offside_position(self, player_pos, defender_line, ball_pos):
        """Check if player is in offside position"""
        if player_pos[0] < ball_pos[0]:  # Behind ball
            return False
        return player_pos[0] > defender_line
    
    def detect_goal(self, ball_center, ball_radius, goal_side='right'):
        """Goal-line technology implementation"""
        goal_line_x = self.goal_coords[goal_side][0]
        goal_y_min = self.goal_coords[goal_side][1]
        goal_y_max = self.goal_coords[goal_side][2]
        
        # Check if ball fully crossed goal line
        ball_edge = ball_center[0] + ball_radius if goal_side == 'right' else ball_center[0] - ball_radius
        
        fully_crossed = (ball_edge > goal_line_x if goal_side == 'right' else ball_edge < goal_line_x)
        within_goal = goal_y_min <= ball_center[1] <= goal_y_max
        
        return {
            'goal_scored': fully_crossed and within_goal,
            'ball_position': ball_center,
            'crossing_distance': abs(ball_edge - goal_line_x),
            'confidence': 0.95 if fully_crossed and within_goal else 0.0
        }
    
    def detect_fouls(self, players, ball_pos, pose_data):
        """Detect fouls and handballs using pose estimation"""
        fouls = []
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players[i+1:], i+1):
                if player1['team'] == player2['team']:
                    continue
                    
                # Check for contact and sudden deceleration
                distance = np.linalg.norm(np.array(player1['position']) - np.array(player2['position']))
                
                if distance < 2.0:  # Close proximity
                    # Check for sudden deceleration (fall detection)
                    if self._detect_fall(player1, player2):
                        fouls.append({
                            'type': 'contact_foul',
                            'players': [player1['id'], player2['id']],
                            'position': player1['position'],
                            'confidence': 0.75
                        })
        
        # Handball detection
        handballs = self._detect_handball(players, ball_pos, pose_data)
        fouls.extend(handballs)
        
        return fouls
    
    def _detect_fall(self, player1, player2):
        """Detect fall posture from pose data"""
        # Simplified fall detection based on velocity change
        return abs(player1.get('velocity', 0) - player2.get('velocity', 0)) > 3.0
    
    def _detect_handball(self, players, ball_pos, pose_data):
        """Detect handball violations"""
        handballs = []
        
        for player in players:
            if player['id'] not in pose_data:
                continue
                
            pose = pose_data[player['id']]
            
            # Check ball-arm intersection
            if self._ball_arm_intersection(ball_pos, pose):
                # Check if arm position is natural
                if not self._is_natural_arm_position(pose):
                    handballs.append({
                        'type': 'handball',
                        'player': player['id'],
                        'position': ball_pos,
                        'confidence': 0.80
                    })
        
        return handballs
    
    def _ball_arm_intersection(self, ball_pos, pose):
        """Check if ball intersects with arm"""
        # Simplified arm detection using shoulder and wrist keypoints
        if 'left_shoulder' in pose and 'left_wrist' in pose:
            arm_distance = np.linalg.norm(np.array(ball_pos) - np.array(pose['left_wrist']))
            return arm_distance < 15.0
        return False
    
    def _is_natural_arm_position(self, pose):
        """Check if arm position is natural"""
        # Simplified natural position check
        if 'left_shoulder' in pose and 'left_wrist' in pose:
            arm_angle = self._calculate_arm_angle(pose['left_shoulder'], pose['left_wrist'])
            return -30 <= arm_angle <= 30  # Natural range
        return True
    
    def _calculate_arm_angle(self, shoulder, wrist):
        """Calculate arm angle from vertical"""
        vector = np.array(wrist) - np.array(shoulder)
        angle = math.degrees(math.atan2(vector[0], vector[1]))
        return angle
    
    def analyze_pressing_lines(self, players):
        """Analyze team pressing and defensive lines"""
        team1_players = [p for p in players if p['team'] == 'team1']
        team2_players = [p for p in players if p['team'] == 'team2']
        
        # Voronoi control analysis
        all_positions = [p['position'] for p in players]
        if len(all_positions) < 4:
            return None
            
        voronoi = Voronoi(all_positions)
        
        # Calculate team control areas
        team1_control = self._calculate_team_control(team1_players, voronoi)
        team2_control = self._calculate_team_control(team2_players, voronoi)
        
        # Defensive line analysis
        defensive_line_break = self._detect_defensive_line_break(team1_players, team2_players)
        
        return {
            'team1_control': team1_control,
            'team2_control': team2_control,
            'defensive_line_break': defensive_line_break,
            'compactness': self._calculate_team_compactness(team1_players)
        }
    
    def _calculate_team_control(self, team_players, voronoi):
        """Calculate team's pitch control percentage"""
        # Simplified control calculation
        team_positions = [p['position'] for p in team_players]
        if not team_positions:
            return 0.0
        
        # Calculate average position spread
        positions_array = np.array(team_positions)
        control_area = np.var(positions_array[:, 0]) * np.var(positions_array[:, 1])
        return min(control_area / 1000.0, 1.0)  # Normalize to 0-1
    
    def _detect_defensive_line_break(self, attacking_team, defending_team):
        """Detect when attacker breaks defensive line with ball control"""
        if not defending_team:
            return False
            
        # Find defensive line (convex hull)
        defender_positions = [p['position'] for p in defending_team]
        if len(defender_positions) < 3:
            return False
            
        hull = ConvexHull(defender_positions)
        
        # Check if any attacker with ball control crosses the line
        for attacker in attacking_team:
            if attacker.get('has_ball', False):
                # Simplified line break detection
                attacker_x = attacker['position'][0]
                max_defender_x = max(pos[0] for pos in defender_positions)
                return attacker_x > max_defender_x
        
        return False
    
    def _calculate_team_compactness(self, team_players):
        """Calculate team compactness metric"""
        if len(team_players) < 2:
            return 0.0
            
        positions = np.array([p['position'] for p in team_players])
        distances = cdist(positions, positions)
        avg_distance = np.mean(distances[distances > 0])
        
        # Normalize compactness (lower distance = higher compactness)
        return max(0, 1 - (avg_distance / 50.0))
    
    def simulate_correct_pass(self, current_pass, players, field_model):
        """What-If: Simulate optimal pass alternative"""
        passer = current_pass['passer']
        current_target = current_pass['target']
        
        # Find optimal teammate
        teammates = [p for p in players if p['team'] == passer['team'] and p['id'] != passer['id']]
        
        best_option = None
        best_score = -1
        
        for teammate in teammates:
            # Calculate pass success probability
            pass_score = self._calculate_pass_score(passer['position'], teammate['position'], players)
            
            if pass_score > best_score:
                best_score = pass_score
                best_option = teammate
        
        if not best_option:
            return None
        
        # Simulate ball trajectory
        trajectory = self._simulate_ball_trajectory(passer['position'], best_option['position'])
        
        # Calculate xT/xG delta
        current_xt = self._calculate_threat_value(current_target['position'])
        optimal_xt = self._calculate_threat_value(best_option['position'])
        
        return {
            'optimal_target': best_option,
            'trajectory': trajectory,
            'pass_success_prob': best_score,
            'xt_improvement': optimal_xt - current_xt,
            'confidence': 0.85
        }
    
    def _calculate_pass_score(self, passer_pos, target_pos, all_players):
        """Calculate pass success probability"""
        distance = np.linalg.norm(np.array(target_pos) - np.array(passer_pos))
        
        # Distance penalty
        distance_score = max(0, 1 - (distance / 50.0))
        
        # Lane openness (simplified)
        opponents = [p for p in all_players if p.get('team') != 'attacking']
        lane_score = self._calculate_lane_openness(passer_pos, target_pos, opponents)
        
        return (distance_score + lane_score) / 2.0
    
    def _calculate_lane_openness(self, start_pos, end_pos, opponents):
        """Calculate how open the passing lane is"""
        if not opponents:
            return 1.0
            
        # Check for opponents in passing lane
        lane_vector = np.array(end_pos) - np.array(start_pos)
        lane_length = np.linalg.norm(lane_vector)
        
        if lane_length == 0:
            return 0.0
            
        lane_unit = lane_vector / lane_length
        
        blocking_opponents = 0
        for opponent in opponents:
            # Project opponent position onto passing lane
            to_opponent = np.array(opponent['position']) - np.array(start_pos)
            projection = np.dot(to_opponent, lane_unit)
            
            if 0 <= projection <= lane_length:
                # Check perpendicular distance to lane
                perp_distance = np.linalg.norm(to_opponent - projection * lane_unit)
                if perp_distance < 3.0:  # Within blocking distance
                    blocking_opponents += 1
        
        return max(0, 1 - (blocking_opponents * 0.3))
    
    def _simulate_ball_trajectory(self, start_pos, end_pos):
        """Simulate ball trajectory with physics"""
        # Simplified trajectory simulation
        trajectory_points = []
        steps = 20
        
        for i in range(steps + 1):
            t = i / steps
            # Parabolic trajectory
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            z = 4 * t * (1 - t)  # Parabolic height
            
            trajectory_points.append([x, y, z])
        
        return trajectory_points
    
    def _calculate_threat_value(self, position):
        """Calculate Expected Threat (xT) value for position"""
        # Simplified xT calculation based on distance to goal
        goal_distance = np.linalg.norm(np.array(position) - np.array([105, 34]))
        return max(0, 1 - (goal_distance / 75.0))