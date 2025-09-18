import numpy as np
import networkx as nx
from collections import defaultdict

# Real Ball Ownership & Pass Detection
class BallOwnershipTracker:
    def __init__(self, fps=30):
        self.fps = fps
        self.current_owner = None
        self.last_owner = None
        self.ownership_history = []
        self.possession_start_frame = None
        self.team_possession_time = defaultdict(float)
        
    def update_ownership(self, ball_pos, players, frame_num):
        if not ball_pos or not players:
            return None
            
        # Find closest player to ball
        min_dist = float('inf')
        closest_player = None
        for player in players:
            px, py = (player['bbox'][0] + player['bbox'][2]) / 2, (player['bbox'][1] + player['bbox'][3]) / 2
            dist = np.sqrt((ball_pos[0] - px)**2 + (ball_pos[1] - py)**2)
            if dist < min_dist and dist < 40:  # 40 pixel threshold
                min_dist = dist
                closest_player = player
        
        if closest_player:
            new_owner = closest_player['id']
            new_team = closest_player['team']
            
            # Check for ownership change
            if self.current_owner != new_owner:
                # End previous possession
                if self.current_owner and self.possession_start_frame:
                    duration = (frame_num - self.possession_start_frame) / self.fps
                    prev_team = next((p['team'] for p in players if p['id'] == self.current_owner), None)
                    if prev_team:
                        self.team_possession_time[prev_team] += duration
                
                # Start new possession
                self.last_owner = self.current_owner
                self.current_owner = new_owner
                self.possession_start_frame = frame_num
                
                # Detect pass or turnover
                if self.last_owner:
                    last_team = next((p['team'] for p in players if p['id'] == self.last_owner), None)
                    if last_team == new_team:
                        return {'type': 'pass', 'from': self.last_owner, 'to': new_owner, 'team': new_team}
                    else:
                        return {'type': 'turnover', 'from': self.last_owner, 'to': new_owner, 'from_team': last_team, 'to_team': new_team}
        
        return None
    
    def get_possession_stats(self):
        total = sum(self.team_possession_time.values())
        if total == 0:
            return {}
        return {team: (time/total)*100 for team, time in self.team_possession_time.items()}

# Pass Network with NetworkX
class PassNetworkAnalyzer:
    def __init__(self):
        self.passes = defaultdict(lambda: defaultdict(int))
        self.player_positions = defaultdict(list)
        self.graph = nx.DiGraph()
        
    def add_pass(self, from_player, to_player, from_pos, to_pos):
        self.passes[from_player][to_player] += 1
        self.player_positions[from_player].append(from_pos)
        self.player_positions[to_player].append(to_pos)
        
        # Update NetworkX graph
        if self.graph.has_edge(from_player, to_player):
            self.graph[from_player][to_player]['weight'] += 1
        else:
            self.graph.add_edge(from_player, to_player, weight=1)
    
    def get_network_graph(self):
        return self.graph
    
    def get_adjacency_matrix(self):
        return dict(self.passes)
    
    def get_player_centrality(self):
        if len(self.graph.nodes()) == 0:
            return {}
        return nx.betweenness_centrality(self.graph, weight='weight')
    
    def get_average_positions(self):
        avg_pos = {}
        for player, positions in self.player_positions.items():
            if positions:
                avg_pos[player] = (np.mean([p[0] for p in positions]), np.mean([p[1] for p in positions]))
        return avg_pos

# Enhanced Tactical KPIs with Real Models
class TacticalKPICalculator:
    def __init__(self):
        self.defensive_actions = []
        self.passes_allowed = 0
        self.buildup_sequences = []
        self.defensive_lines = defaultdict(list)
        self.xt_grid = self._create_xt_grid()
        
    def _create_xt_grid(self):
        # Simplified xT grid (12x8 zones)
        grid = np.array([
            [0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.01, 0.00],
            [0.00, 0.01, 0.02, 0.03, 0.03, 0.02, 0.01, 0.00],
            [0.01, 0.02, 0.04, 0.06, 0.06, 0.04, 0.02, 0.01],
            [0.01, 0.03, 0.06, 0.10, 0.10, 0.06, 0.03, 0.01],
            [0.02, 0.04, 0.08, 0.15, 0.15, 0.08, 0.04, 0.02],
            [0.03, 0.06, 0.12, 0.25, 0.25, 0.12, 0.06, 0.03],
            [0.04, 0.08, 0.16, 0.35, 0.35, 0.16, 0.08, 0.04],
            [0.06, 0.12, 0.24, 0.50, 0.50, 0.24, 0.12, 0.06],
            [0.08, 0.16, 0.32, 0.65, 0.65, 0.32, 0.16, 0.08],
            [0.10, 0.20, 0.40, 0.80, 0.80, 0.40, 0.20, 0.10],
            [0.12, 0.24, 0.48, 0.95, 0.95, 0.48, 0.24, 0.12],
            [0.15, 0.30, 0.60, 1.00, 1.00, 0.60, 0.30, 0.15]
        ])
        return grid
    
    def calculate_xt(self, from_pos, to_pos, field_dims=(105, 68)):
        # Convert positions to grid coordinates
        from_x = int((from_pos[0] / field_dims[0]) * 11)
        from_y = int((from_pos[1] / field_dims[1]) * 7)
        to_x = int((to_pos[0] / field_dims[0]) * 11)
        to_y = int((to_pos[1] / field_dims[1]) * 7)
        
        # Clamp to grid bounds
        from_x, from_y = max(0, min(11, from_x)), max(0, min(7, from_y))
        to_x, to_y = max(0, min(11, to_x)), max(0, min(7, to_y))
        
        return self.xt_grid[to_x, to_y] - self.xt_grid[from_x, from_y]
    
    def calculate_ppda(self):
        if len(self.defensive_actions) == 0:
            return 0
        return self.passes_allowed / len(self.defensive_actions)
    
    def detect_buildup_sequence(self, passes, team):
        if len(passes) >= 4:
            # Check field progression
            start_y = passes[0]['from_pos'][1]
            end_y = passes[-1]['to_pos'][1]
            progression = abs(end_y - start_y)
            
            if progression > 25:  # Significant field progression
                self.buildup_sequences.append({
                    'team': team,
                    'passes': len(passes),
                    'progression': progression,
                    'start_zone': self._get_zone(passes[0]['from_pos']),
                    'end_zone': self._get_zone(passes[-1]['to_pos'])
                })
                return True
        return False
    
    def track_defensive_line(self, players, team, frame_num):
        defenders = [p for p in players if p['team'] == team]
        if len(defenders) >= 4:
            y_positions = [(p['bbox'][1] + p['bbox'][3]) / 2 for p in defenders]
            avg_line = np.mean(y_positions)
            self.defensive_lines[team].append((frame_num, avg_line))
            return avg_line
        return None
    
    def _get_zone(self, pos):
        if pos[1] < 22.67:  # Defensive third
            return 'defensive'
        elif pos[1] < 45.33:  # Middle third
            return 'middle'
        else:  # Final third
            return 'final'
    
    def get_kpi_summary(self):
        return {
            'ppda': self.calculate_ppda(),
            'buildup_sequences': len(self.buildup_sequences),
            'avg_defensive_line': {team: np.mean([line[1] for line in lines]) for team, lines in self.defensive_lines.items()},
            'defensive_actions': len(self.defensive_actions),
            'total_passes_allowed': self.passes_allowed
        }