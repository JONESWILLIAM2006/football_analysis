# Enhanced Player Overlays and Tactical Visualizations
import cv2
import numpy as np
from scipy.spatial import Voronoi
import plotly.graph_objects as go
import streamlit as st

class PlayerOverlayEngine:
    def __init__(self):
        self.team_colors = {
            'team1': (255, 0, 0),    # Red
            'team2': (0, 0, 255),    # Blue
            'referee': (255, 255, 0) # Yellow
        }
        self.jersey_numbers = {}
        self.reid_embeddings = {}
        
    def draw_player_overlay(self, frame, player_id, bbox, team, jersey_num=None):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = self.team_colors.get(team, (255, 255, 255))
        
        # Enhanced bounding box with team colors
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Player ID tag with background
        tag_text = f"P{player_id}"
        if jersey_num:
            tag_text = f"#{jersey_num}"
            
        tag_size = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (x1, y1-30), (x1+tag_size[0]+10, y1), color, -1)
        cv2.putText(frame, tag_text, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Team indicator dot
        cv2.circle(frame, (x2-15, y1+15), 8, color, -1)
        
        return frame
    
    def extract_reid_features(self, frame, bbox):
        """Extract ReID features for consistent tracking"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return np.zeros(128)
        
        # Simple color histogram as ReID feature
        crop_resized = cv2.resize(crop, (64, 128))
        hist = cv2.calcHist([crop_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return hist.flatten() / (np.linalg.norm(hist.flatten()) + 1e-6)

class PitchControlVisualizer:
    def __init__(self, pitch_width=105, pitch_height=68):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        
    def compute_voronoi_control(self, team1_positions, team2_positions):
        """Compute pitch control using Voronoi diagrams"""
        all_positions = list(team1_positions.values()) + list(team2_positions.values())
        team_labels = ['team1'] * len(team1_positions) + ['team2'] * len(team2_positions)
        
        if len(all_positions) < 4:
            return None, None
        
        try:
            vor = Voronoi(all_positions)
            return vor, team_labels
        except:
            return None, None
    
    def create_control_heatmap(self, team1_positions, team2_positions, frame_shape):
        """Create pitch control heatmap"""
        height, width = frame_shape[:2]
        control_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create grid for influence calculation
        y_coords, x_coords = np.mgrid[0:height:10, 0:width:10]
        grid_points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        for point in grid_points:
            # Calculate influence from each team
            team1_influence = self._calculate_team_influence(point, team1_positions)
            team2_influence = self._calculate_team_influence(point, team2_positions)
            
            # Determine controlling team
            if team1_influence > team2_influence:
                intensity = min(255, int(team1_influence * 100))
                control_map[int(point[1]):int(point[1])+10, int(point[0]):int(point[0])+10] = [intensity, 0, 0]
            else:
                intensity = min(255, int(team2_influence * 100))
                control_map[int(point[1]):int(point[1])+10, int(point[0]):int(point[0])+10] = [0, 0, intensity]
        
        return control_map
    
    def _calculate_team_influence(self, point, team_positions):
        """Calculate team influence at a point"""
        if not team_positions:
            return 0
        
        total_influence = 0
        for pos in team_positions.values():
            distance = np.linalg.norm(np.array(point) - np.array(pos))
            influence = 1 / (1 + distance / 50)  # Influence decreases with distance
            total_influence += influence
        
        return total_influence / len(team_positions)

class TacticalMapOverlay:
    def __init__(self):
        self.mini_pitch_size = (200, 130)
        
    def create_mini_pitch(self, team1_positions, team2_positions, ball_position=None):
        """Create tactical mini-pitch overlay"""
        width, height = self.mini_pitch_size
        mini_pitch = np.ones((height, width, 3), dtype=np.uint8) * 34  # Green background
        
        # Draw pitch lines
        cv2.rectangle(mini_pitch, (10, 10), (width-10, height-10), (255, 255, 255), 2)
        cv2.line(mini_pitch, (width//2, 10), (width//2, height-10), (255, 255, 255), 1)
        cv2.circle(mini_pitch, (width//2, height//2), 20, (255, 255, 255), 1)
        
        # Draw players
        for pos in team1_positions.values():
            x = int((pos[0] / 105) * (width - 20) + 10)
            y = int((pos[1] / 68) * (height - 20) + 10)
            cv2.circle(mini_pitch, (x, y), 4, (255, 0, 0), -1)
        
        for pos in team2_positions.values():
            x = int((pos[0] / 105) * (width - 20) + 10)
            y = int((pos[1] / 68) * (height - 20) + 10)
            cv2.circle(mini_pitch, (x, y), 4, (0, 0, 255), -1)
        
        # Draw ball
        if ball_position:
            x = int((ball_position[0] / 105) * (width - 20) + 10)
            y = int((ball_position[1] / 68) * (height - 20) + 10)
            cv2.circle(mini_pitch, (x, y), 3, (255, 255, 255), -1)
        
        return mini_pitch
    
    def draw_passing_lanes(self, frame, current_player_pos, potential_targets, optimal_target=None):
        """Draw passing lanes and highlight optimal pass"""
        for target_pos in potential_targets:
            # Draw potential pass lane
            cv2.arrowedLine(frame, 
                          tuple(map(int, current_player_pos)), 
                          tuple(map(int, target_pos)), 
                          (255, 255, 0), 2, tipLength=0.3)
        
        # Highlight optimal pass
        if optimal_target:
            cv2.arrowedLine(frame, 
                          tuple(map(int, current_player_pos)), 
                          tuple(map(int, optimal_target)), 
                          (0, 255, 0), 4, tipLength=0.3)
            cv2.putText(frame, "OPTIMAL", 
                       tuple(map(int, optimal_target)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

class SpacePressureAnalyzer:
    def __init__(self):
        self.pressure_zones = {}
        
    def analyze_pressing_zones(self, team_positions, ball_position):
        """Analyze where team applies pressure"""
        if not ball_position:
            return {}
        
        pressing_zones = {}
        
        # Define pressure zones around ball
        for zone_name, (min_dist, max_dist) in [('high', (0, 15)), ('medium', (15, 30)), ('low', (30, 50))]:
            players_in_zone = 0
            for pos in team_positions.values():
                distance = np.linalg.norm(np.array(pos) - np.array(ball_position))
                if min_dist <= distance <= max_dist:
                    players_in_zone += 1
            
            pressing_zones[zone_name] = players_in_zone
        
        return pressing_zones
    
    def create_space_heatmap(self, all_positions, frame_shape):
        """Create heatmap showing available space"""
        height, width = frame_shape[:2]
        space_map = np.ones((height, width), dtype=np.float32)
        
        # Calculate space availability
        for y in range(0, height, 10):
            for x in range(0, width, 10):
                min_distance = float('inf')
                for pos in all_positions:
                    distance = np.linalg.norm(np.array([x, y]) - np.array(pos))
                    min_distance = min(min_distance, distance)
                
                space_map[y:y+10, x:x+10] = min(1.0, min_distance / 100)
        
        return space_map

class EventOverlayEngine:
    def __init__(self):
        self.flash_duration = 30  # frames
        self.flash_counter = 0
        self.current_event = None
        
    def trigger_turnover_flash(self, event_data):
        """Trigger red flash for turnovers"""
        self.current_event = event_data
        self.flash_counter = self.flash_duration
    
    def draw_event_overlay(self, frame, event_type, position, alternative_pass=None):
        """Draw event overlays on frame"""
        if self.flash_counter > 0:
            # Red flash overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Event text
            cv2.putText(frame, "TURNOVER!", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            self.flash_counter -= 1
        
        # Draw alternative pass suggestion
        if alternative_pass:
            from_pos, to_pos = alternative_pass
            cv2.arrowedLine(frame, tuple(map(int, from_pos)), tuple(map(int, to_pos)), 
                          (0, 255, 255), 3, tipLength=0.3)
            cv2.putText(frame, "BETTER OPTION", tuple(map(int, to_pos)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame

class InteractiveUI:
    def __init__(self):
        self.selected_player = None
        self.timeline_position = 0
        
    def create_overlay_controls(self):
        """Create UI controls for overlays"""
        st.sidebar.subheader("🎨 Visual Overlays")
        
        overlay_settings = {}
        overlay_settings['player_tags'] = st.sidebar.checkbox("Player Tags & IDs", value=True)
        overlay_settings['pitch_control'] = st.sidebar.checkbox("Pitch Control Heatmap", value=False)
        overlay_settings['tactical_map'] = st.sidebar.checkbox("Tactical Mini-Map", value=False)
        overlay_settings['pressure_zones'] = st.sidebar.checkbox("Pressure Zones", value=False)
        overlay_settings['passing_lanes'] = st.sidebar.checkbox("Passing Lanes", value=False)
        
        return overlay_settings
    
    def create_player_stats_card(self, player_id, stats):
        """Create interactive player stats card"""
        with st.expander(f"Player {player_id} Stats"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Passes", stats.get('passes', 0))
                st.metric("Pass Accuracy", f"{stats.get('pass_accuracy', 0):.1f}%")
            
            with col2:
                st.metric("Distance", f"{stats.get('distance_covered', 0):.1f}m")
                st.metric("Sprints", stats.get('sprints', 0))
            
            with col3:
                st.metric("xG Contribution", f"{stats.get('xg_contribution', 0):.2f}")
                st.metric("Touches", stats.get('touches', 0))
    
    def create_timeline_scrubber(self, total_frames, events):
        """Create timeline with event markers"""
        st.subheader("📅 Match Timeline")
        
        # Timeline slider
        current_frame = st.slider("Timeline", 0, total_frames, 0)
        
        # Event chips
        if events:
            st.write("**Key Events:**")
            event_cols = st.columns(min(len(events), 5))
            
            for i, event in enumerate(events[:5]):
                with event_cols[i]:
                    if st.button(f"{event['type']} - {event['frame']//30}s", key=f"event_{i}"):
                        st.session_state.timeline_jump = event['frame']
        
        return current_frame
    
    def create_plotly_pitch_control(self, team1_positions, team2_positions):
        """Create interactive pitch control visualization with Plotly"""
        fig = go.Figure()
        
        # Draw pitch
        fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, 
                     line=dict(color="white", width=2), fillcolor="green", opacity=0.3)
        
        # Center circle
        fig.add_shape(type="circle", x0=52.5-9.15, y0=34-9.15, x1=52.5+9.15, y1=34+9.15,
                     line=dict(color="white", width=2))
        
        # Team 1 players
        if team1_positions:
            x1_coords = [pos[0] for pos in team1_positions.values()]
            y1_coords = [pos[1] for pos in team1_positions.values()]
            fig.add_trace(go.Scatter(x=x1_coords, y=y1_coords, mode='markers',
                                   marker=dict(size=15, color='red'), name='Team 1'))
        
        # Team 2 players
        if team2_positions:
            x2_coords = [pos[0] for pos in team2_positions.values()]
            y2_coords = [pos[1] for pos in team2_positions.values()]
            fig.add_trace(go.Scatter(x=x2_coords, y=y2_coords, mode='markers',
                                   marker=dict(size=15, color='blue'), name='Team 2'))
        
        fig.update_layout(
            title="Interactive Pitch Control",
            xaxis=dict(range=[0, 105], title="Length (m)"),
            yaxis=dict(range=[0, 68], title="Width (m)"),
            showlegend=True,
            height=400
        )
        
        return fig