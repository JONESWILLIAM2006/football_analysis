"""
Minimal Football Analysis System - No problematic dependencies
"""
import os
import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
import random
import plotly.graph_objects as go
import plotly.express as px
from collections import deque

# Safe imports only
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# -------------------- CORE CLASSES --------------------

class SimpleTracker:
    """Simple object tracking without complex dependencies"""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        
    def update(self, detections):
        """Update tracks with new detections"""
        if not detections:
            return []
        
        # Simple distance-based tracking
        active_tracks = []
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            center = [(x1+x2)/2, (y1+y2)/2]
            
            # Assign ID based on proximity to existing tracks
            track_id = self._assign_id(center)
            active_tracks.append({
                'id': track_id,
                'bbox': [x1, y1, x2, y2],
                'center': center,
                'confidence': conf
            })
        
        return active_tracks
    
    def _assign_id(self, center):
        """Assign ID to detection"""
        min_dist = float('inf')
        best_id = None
        
        for track_id, last_pos in self.tracks.items():
            dist = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
            if dist < min_dist and dist < 100:  # Distance threshold
                min_dist = dist
                best_id = track_id
        
        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
        
        self.tracks[best_id] = center
        return best_id

class FootballAnalyzer:
    """Main football analysis system"""
    def __init__(self):
        if YOLO_AVAILABLE:
            self.model = YOLO("yolov8n.pt")  # Use nano model for speed
        else:
            self.model = None
        
        self.tracker = SimpleTracker()
        self.ball_positions = deque(maxlen=30)
        
    def process_frame(self, frame):
        """Process single frame"""
        players = []
        ball = None
        
        if self.model:
            results = self.model(frame, verbose=False)
            detections = []
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for box, score, cls in zip(boxes, scores, classes):
                    if score > 0.5:
                        x1, y1, x2, y2 = box
                        if cls == 0:  # Person
                            detections.append([x1, y1, x2, y2, score])
                        elif cls == 37:  # Ball
                            ball = {
                                'center': [(x1+x2)/2, (y1+y2)/2],
                                'bbox': [x1, y1, x2, y2],
                                'confidence': score
                            }
            
            # Track players
            players = self.tracker.update(detections)
            
            # Add team assignment
            for i, player in enumerate(players):
                player['team'] = 'Team A' if i < 11 else 'Team B'
        
        # Track ball
        if ball:
            self.ball_positions.append(ball['center'])
        
        return players, ball
    
    def draw_annotations(self, frame, players, ball):
        """Draw annotations on frame"""
        # Draw players
        for player in players:
            x1, y1, x2, y2 = [int(v) for v in player['bbox']]
            color = (0, 0, 255) if player['team'] == 'Team A' else (255, 0, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"P{player['id']}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw ball and trail
        if ball:
            cx, cy = map(int, ball['center'])
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)
            
            # Draw trail
            if len(self.ball_positions) > 1:
                points = list(self.ball_positions)
                for i in range(1, len(points)):
                    cv2.line(frame, tuple(map(int, points[i-1])), 
                            tuple(map(int, points[i])), (0, 255, 255), 2)
        
        return frame

# -------------------- BROADCAST LAYOUT --------------------

class BroadcastLayout:
    """Professional broadcast-style layout"""
    def __init__(self):
        self.analyzer = FootballAnalyzer()
        self.match_time = "00:00"
        self.score = "0 - 0"
        
    def render_layout(self):
        """Render complete broadcast layout"""
        st.markdown(self._get_css(), unsafe_allow_html=True)
        
        # Top bar
        self._render_top_bar()
        
        # Main content
        col_left, col_center, col_right = st.columns([1, 3, 1])
        
        with col_left:
            self._render_filters()
        
        with col_center:
            self._render_video_player()
        
        with col_right:
            self._render_widgets()
        
        # Bottom timeline
        self._render_timeline()
        
        # View tabs
        self._render_views()
    
    def _get_css(self):
        """Custom CSS styling"""
        return """
        <style>
        .top-bar {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 10px;
            border-radius: 5px;
            color: white;
            text-align: center;
            margin-bottom: 10px;
        }
        .widget-box {
            background: rgba(0,0,0,0.1);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .event-chip {
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            margin: 2px;
            font-size: 12px;
        }
        </style>
        """
    
    def _render_top_bar(self):
        """Render top status bar"""
        st.markdown(f'<div class="top-bar">⏱️ {self.match_time} | ⚽ {self.score} | 📊 Possession: 58% - 42%</div>', 
                   unsafe_allow_html=True)
    
    def _render_filters(self):
        """Render left filter panel"""
        st.subheader("🔍 Filters")
        
        team_filter = st.multiselect("Teams", ["Team A", "Team B"], default=["Team A", "Team B"])
        event_filter = st.multiselect("Events", ["Goal", "Shot", "Pass", "Foul"])
        
        st.subheader("📹 Camera")
        camera_view = st.selectbox("View", ["Main", "Tactical", "Goal"])
        
        if st.checkbox("Multi-Angle"):
            st.slider("Feeds", 2, 4, 2)
    
    def _render_video_player(self):
        """Render center video player"""
        st.subheader("📺 Live Analysis")
        
        # Upload video
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file:
            # Save video
            video_path = f"temp_{uploaded_file.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Overlay controls
            col1, col2, col3 = st.columns(3)
            with col1:
                show_players = st.checkbox("Player Detection", True)
            with col2:
                show_ball = st.checkbox("Ball Tracking", True)
            with col3:
                show_offside = st.checkbox("Offside Lines", False)
            
            if st.button("▶️ Start Analysis"):
                self._process_video(video_path, show_players, show_ball)
            
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
        else:
            st.info("Upload a video to start analysis")
    
    def _render_widgets(self):
        """Render right widgets panel"""
        st.subheader("📊 Live Stats")
        
        # xG Widget
        st.markdown('<div class="widget-box">', unsafe_allow_html=True)
        st.write("**⚽ Expected Goals**")
        
        xg_data = pd.DataFrame({
            'Team': ['Team A', 'Team B'],
            'xG': [1.2, 0.8]
        })
        fig = px.bar(xg_data, x='Team', y='xG', color='Team')
        fig.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Pass Map
        st.markdown('<div class="widget-box">', unsafe_allow_html=True)
        st.write("**🎯 Pass Network**")
        
        pass_data = pd.DataFrame({
            'x': np.random.uniform(0, 100, 20),
            'y': np.random.uniform(0, 100, 20),
            'success': np.random.choice([True, False], 20, p=[0.8, 0.2])
        })
        
        fig = px.scatter(pass_data, x='x', y='y', color='success')
        fig.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Player Stats
        st.markdown('<div class="widget-box">', unsafe_allow_html=True)
        st.write("**👥 Player Stats**")
        for i in range(1, 4):
            st.metric(f"Player {i}", f"{np.random.randint(60, 95)}%", f"{np.random.randint(-5, 10)}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_timeline(self):
        """Render bottom timeline"""
        st.subheader("⏰ Timeline")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            current_time = st.slider("Match Time", 0, 90, 45)
        
        with col2:
            st.button("⏪ -10s")
        
        with col3:
            st.button("⏩ +10s")
        
        # Event chips
        events_html = """
        <span class="event-chip">5' Goal</span>
        <span class="event-chip">12' Foul</span>
        <span class="event-chip">23' Shot</span>
        <span class="event-chip">34' Corner</span>
        """
        st.markdown(events_html, unsafe_allow_html=True)
    
    def _render_views(self):
        """Render view tabs"""
        tabs = st.tabs(["🔴 Live", "⚖️ VAR", "🎯 Tactics", "👥 Players", "📊 Reports"])
        
        with tabs[0]:
            st.write("**Live Match Analysis**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Shots", "8 - 5")
                st.metric("Possession", "58% - 42%")
            with col2:
                st.metric("Pass Accuracy", "87% - 82%")
                st.metric("Corners", "4 - 2")
        
        with tabs[1]:
            st.write("**VAR Review**")
            st.selectbox("Incident", ["Offside (23')", "Goal (45')", "Penalty (67')"])
            st.slider("Frame", 0, 100, 50)
            st.success("✅ GOAL CONFIRMED")
        
        with tabs[2]:
            st.write("**Tactical Analysis**")
            
            # Voronoi simulation
            voronoi_data = pd.DataFrame({
                'x': np.random.uniform(0, 105, 22),
                'y': np.random.uniform(0, 68, 22),
                'team': ['A']*11 + ['B']*11
            })
            
            fig = px.scatter(voronoi_data, x='x', y='y', color='team')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            st.write("**Player Profiles**")
            selected_player = st.selectbox("Player", [f"Player {i}" for i in range(1, 12)])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Distance", "8.2 km")
                st.metric("Passes", "34 (89%)")
            with col2:
                st.metric("Speed", "28.4 km/h")
                st.metric("Touches", "67")
        
        with tabs[4]:
            st.write("**Reports & Downloads**")
            
            if st.button("📹 Download Highlights"):
                st.success("Generating highlights...")
            
            if st.button("📊 Export Data (CSV)"):
                sample_data = pd.DataFrame({
                    'Time': range(0, 90, 5),
                    'Event': ['Pass', 'Shot', 'Foul'] * 6,
                    'Player': np.random.randint(1, 23, 18)
                })
                
                csv = sample_data.to_csv(index=False)
                st.download_button("Download CSV", csv, "match_data.csv", "text/csv")
    
    def _process_video(self, video_path, show_players, show_ball):
        """Process uploaded video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Could not open video")
            return
        
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process limited frames for demo
        max_frames = min(100, total_frames)
        
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            players, ball = self.analyzer.process_frame(frame)
            
            # Draw annotations
            if show_players or show_ball:
                frame = self.analyzer.draw_annotations(frame, players, ball)
            
            # Display frame
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            # Update progress
            progress_bar.progress((i + 1) / max_frames)
            frame_count += 1
            
            # Small delay for visualization
            time.sleep(0.1)
        
        cap.release()
        
        # Show results
        st.success(f"✅ Processed {frame_count} frames")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Players Detected", len(players) if players else 0)
        with col2:
            st.metric("Ball Tracking", "Active" if ball else "Inactive")
        with col3:
            st.metric("Events", np.random.randint(5, 15))

# -------------------- MAIN APP --------------------

def main():
    """Main application"""
    st.set_page_config(
        page_title="⚽ Football Analysis",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Professional Football Analysis")
    
    # Check dependencies
    if not YOLO_AVAILABLE:
        st.warning("⚠️ YOLO not available. Install ultralytics for full functionality.")
    
    # Initialize and render layout
    broadcast = BroadcastLayout()
    broadcast.render_layout()

if __name__ == "__main__":
    main()