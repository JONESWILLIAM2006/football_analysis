# Enhanced Football Analysis System - Main Integration
import streamlit as st
import cv2
import numpy as np
from enhanced_detection import MultiObjectTracker, SegmentationDetector
from pitch_homography import PitchHomographyDetector
from tactical_analysis import TacticalEventDetector, GraphPassingNetwork
from multilingual_commentary import MultilingualCommentary, RealTimeHeatmapGenerator, InteractiveControls
from model_optimization import ONNXInference, ModelEnsemble, GPUAccelerator
import time
import mediapipe as mp

class EnhancedFootballAnalysisSystem:
    def __init__(self):
        # Core components
        self.multi_tracker = MultiObjectTracker()
        self.pitch_detector = PitchHomographyDetector()
        self.tactical_analyzer = TacticalEventDetector()
        self.pass_network = GraphPassingNetwork()
        self.commentary_engine = MultilingualCommentary()
        self.heatmap_generator = RealTimeHeatmapGenerator()
        self.interactive_controls = InteractiveControls()
        
        # Optimization components
        self.gpu_accelerator = GPUAccelerator()
        self.model_ensemble = None
        
        # MediaPipe for pose estimation
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Analysis state
        self.frame_count = 0
        self.events = []
        self.player_poses = {}
        self.calibrated = False
        
    def initialize_optimized_models(self, model_paths):
        """Initialize optimized model ensemble"""
        if model_paths:
            self.model_ensemble = ModelEnsemble(model_paths)
            st.success(f"Loaded {len(model_paths)} optimized models")
    
    def process_video_enhanced(self, video_path, language='en', use_pose=True):
        """Enhanced video processing with all improvements"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video file")
            return None
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('outputs/enhanced_analysis.mp4', fourcc, fps, (1280, 720))
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        video_placeholder = st.empty()
        
        # Analysis containers
        events_container = st.container()
        stats_container = st.container()
        
        frame_data_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frame = cv2.resize(frame, (1280, 720))
            
            # Auto-calibrate pitch on first frame
            if not self.calibrated:
                if self.pitch_detector.calibrate_homography(frame):
                    self.calibrated = True
                    st.success("✅ Pitch automatically calibrated!")
            
            # Enhanced detection and tracking
            tracking_results = self._process_frame_enhanced(frame, use_pose)
            
            # Tactical analysis
            tactical_events = self.tactical_analyzer.analyze_frame(tracking_results)
            self.events.extend(tactical_events)
            
            # Update pass network
            self._update_pass_network(tracking_results)
            
            # Update heatmaps
            self._update_heatmaps(tracking_results)
            
            # Generate commentary
            if tactical_events:
                for event in tactical_events:
                    commentary = self.commentary_engine.generate_commentary(event, language)
                    # Store for later playback
                    event['commentary'] = commentary
            
            # Visualize results
            vis_frame = self._visualize_enhanced_results(frame, tracking_results, tactical_events)
            
            # Update UI
            progress = self.frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {self.frame_count}/{total_frames}")
            
            # Show video every 10 frames to avoid lag
            if self.frame_count % 10 == 0:
                video_placeholder.image(vis_frame, channels="BGR", use_container_width=True)
            
            # Update live stats every 30 frames
            if self.frame_count % 30 == 0:
                self._update_live_stats(stats_container)
            
            out.write(vis_frame)
            frame_data_history.append(tracking_results)
        
        cap.release()
        out.release()
        
        # Final analysis
        return self._generate_final_analysis(frame_data_history, language)
    
    def _process_frame_enhanced(self, frame, use_pose=True):
        """Enhanced frame processing with all detection methods"""
        # Multi-object tracking
        tracking_results = self.multi_tracker.update(frame)
        
        # Add pose estimation if enabled
        if use_pose and tracking_results.get('players'):
            for track in tracking_results['players']:
                player_id = track.track_id
                bbox = track.bbox
                
                # Extract player region
                x1, y1, x2, y2 = [int(v) for v in bbox]
                player_crop = frame[y1:y2, x1:x2]
                
                if player_crop.size > 0:
                    # Run pose detection
                    rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose_detector.process(rgb_crop)
                    
                    if pose_results.pose_landmarks:
                        self.player_poses[player_id] = pose_results.pose_landmarks
        
        # Convert to world coordinates if calibrated
        if self.calibrated:
            tracking_results = self._convert_to_world_coordinates(tracking_results)
        
        return tracking_results
    
    def _convert_to_world_coordinates(self, tracking_results):
        """Convert pixel coordinates to world coordinates"""
        if tracking_results.get('players'):
            for track in tracking_results['players']:
                bbox = track.bbox
                center_pixel = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                world_pos = self.pitch_detector.pixel_to_world([center_pixel])
                if world_pos is not None:
                    track.world_position = world_pos[0]
        
        if tracking_results.get('ball'):
            ball_pixel = tracking_results['ball'][:2]
            world_pos = self.pitch_detector.pixel_to_world([ball_pixel])
            if world_pos is not None:
                tracking_results['ball_world'] = world_pos[0]
        
        return tracking_results
    
    def _update_pass_network(self, tracking_results):
        """Update passing network with new data"""
        # Simplified pass detection logic
        if tracking_results.get('ball') and len(tracking_results.get('players', [])) > 1:
            # Find closest players to ball
            ball_pos = tracking_results['ball'][:2]
            
            closest_players = []
            for track in tracking_results['players']:
                bbox = track.bbox
                player_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                distance = np.linalg.norm(np.array(ball_pos) - np.array(player_center))
                closest_players.append((track.track_id, distance, player_center))
            
            closest_players.sort(key=lambda x: x[1])
            
            # If ball changes possession (simplified)
            if len(closest_players) >= 2 and closest_players[0][1] < 50:
                current_player = closest_players[0][0]
                
                # Check if this is a pass (ball was with different player before)
                if hasattr(self, 'last_ball_owner') and self.last_ball_owner != current_player:
                    # Add pass to network
                    from_pos = getattr(self, 'last_ball_pos', closest_players[1][2])
                    to_pos = closest_players[0][2]
                    
                    self.pass_network.add_pass(
                        self.last_ball_owner, 
                        current_player,
                        from_pos,
                        to_pos,
                        success=True  # Simplified
                    )
                
                self.last_ball_owner = current_player
                self.last_ball_pos = closest_players[0][2]
    
    def _update_heatmaps(self, tracking_results):
        """Update player heatmaps"""
        if tracking_results.get('players'):
            for track in tracking_results['players']:
                if hasattr(track, 'world_position'):
                    self.heatmap_generator.update_heatmap(
                        track.track_id, 
                        track.world_position
                    )
    
    def _visualize_enhanced_results(self, frame, tracking_results, events):
        """Enhanced visualization with all analysis results"""
        vis_frame = frame.copy()
        
        # Draw pitch calibration
        if self.calibrated:
            vis_frame = self.pitch_detector.visualize_calibration(vis_frame)
        
        # Draw player tracking
        if tracking_results.get('players'):
            for track in tracking_results['players']:
                bbox = track.bbox
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw player ID
                cv2.putText(vis_frame, f'P{track.track_id}', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw pose if available
                if track.track_id in self.player_poses:
                    self._draw_pose_landmarks(vis_frame, self.player_poses[track.track_id], (x1, y1))
        
        # Draw ball tracking
        if tracking_results.get('ball'):
            ball_pos = tracking_results['ball'][:2]
            cv2.circle(vis_frame, (int(ball_pos[0]), int(ball_pos[1])), 10, (0, 0, 255), -1)
            cv2.putText(vis_frame, 'BALL', 
                       (int(ball_pos[0])-20, int(ball_pos[1])-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw tactical events
        for event in events:
            event_text = f"{event['type']}: {event.get('description', '')}"
            cv2.putText(vis_frame, event_text, (50, 100 + len(events) * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw pass network connections
        self._draw_pass_network(vis_frame, tracking_results)
        
        return vis_frame
    
    def _draw_pose_landmarks(self, frame, landmarks, offset):
        """Draw pose landmarks on frame"""
        if not landmarks:
            return
        
        # Draw key pose points
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE
        ]
        
        for point in key_points:
            landmark = landmarks.landmark[point.value]
            x = int(landmark.x * 100 + offset[0])  # Scale to bbox
            y = int(landmark.y * 100 + offset[1])
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
    
    def _draw_pass_network(self, frame, tracking_results):
        """Draw pass network connections"""
        if not tracking_results.get('players') or len(tracking_results['players']) < 2:
            return
        
        # Get recent passes from network
        recent_passes = list(self.pass_network.pass_history)[-10:]  # Last 10 passes
        
        for pass_data in recent_passes:
            from_pos = pass_data['from_pos']
            to_pos = pass_data['to_pos']
            
            # Draw pass line
            cv2.arrowedLine(frame, 
                           (int(from_pos[0]), int(from_pos[1])),
                           (int(to_pos[0]), int(to_pos[1])),
                           (0, 255, 255), 2, tipLength=0.3)
    
    def _update_live_stats(self, container):
        """Update live statistics display"""
        with container:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Events", len(self.events))
                st.metric("Frames Processed", self.frame_count)
            
            with col2:
                # Pass network stats
                key_players = self.pass_network.get_key_players()
                if key_players:
                    st.write("**Key Players:**")
                    st.write(f"Playmaker: Player {key_players.get('playmaker', 'N/A')}")
                    st.write(f"Connector: Player {key_players.get('connector', 'N/A')}")
            
            with col3:
                # Passing patterns
                patterns = self.pass_network.analyze_passing_patterns()
                if patterns:
                    st.write("**Passing Patterns:**")
                    st.write(f"Short passes: {patterns.get('short_passes', 0)}")
                    st.write(f"Long passes: {patterns.get('long_passes', 0)}")
    
    def _generate_final_analysis(self, frame_data_history, language):
        """Generate comprehensive final analysis"""
        analysis_results = {
            'total_frames': self.frame_count,
            'events': self.events,
            'pass_network': self.pass_network,
            'key_players': self.pass_network.get_key_players(),
            'passing_patterns': self.pass_network.analyze_passing_patterns(),
            'calibrated': self.calibrated
        }
        
        # Generate multilingual summary
        if self.events:
            summary_text = f"Match analysis complete. {len(self.events)} tactical events detected."
            if language != 'en':
                summary_text = self.commentary_engine.translate_text(summary_text, language)
            
            analysis_results['summary'] = summary_text
        
        return analysis_results
    
    def create_interactive_dashboard(self, analysis_results):
        """Create interactive dashboard with all features"""
        st.header("🏆 Enhanced Football Analysis Dashboard")
        
        # Language selection
        language = st.selectbox("Select Language", 
                               list(self.commentary_engine.supported_languages.keys()),
                               format_func=lambda x: self.commentary_engine.supported_languages[x])
        
        # Interactive controls
        st.subheader("🎛️ Interactive Controls")
        
        # Event filtering
        event_types = list(set([event.get('type', 'unknown') for event in analysis_results['events']]))
        selected_events = st.multiselect("Filter Events", event_types, default=event_types)
        self.interactive_controls.set_event_filter(selected_events)
        
        # Time range filtering
        if analysis_results['total_frames'] > 0:
            time_range = st.slider("Time Range (seconds)", 
                                 0, analysis_results['total_frames'] // 30,
                                 (0, analysis_results['total_frames'] // 30))
            self.interactive_controls.set_time_filter(time_range[0] * 30, time_range[1] * 30)
        
        # Filter events
        filtered_events = self.interactive_controls.filter_events(analysis_results['events'])
        
        # Display filtered results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Event Analysis")
            if filtered_events:
                event_df = st.dataframe(filtered_events)
                
                # Event timeline
                timeline_data = self.interactive_controls.get_timeline_data(filtered_events)
                if timeline_data:
                    st.line_chart([item['count'] for item in timeline_data])
        
        with col2:
            st.subheader("🗺️ Player Heatmaps")
            
            # Player selection for heatmap
            available_players = list(self.heatmap_generator.heatmap_data.keys())
            if available_players:
                selected_player = st.selectbox("Select Player", available_players)
                
                # Generate and display heatmap
                heatmap_image = self.heatmap_generator.generate_heatmap_image(selected_player)
                st.image(heatmap_image, caption=f"Player {selected_player} Heatmap")
                
                # Player stats
                player_stats = self.heatmap_generator.get_player_stats(selected_player)
                st.json(player_stats)
        
        # Pass Network Visualization
        st.subheader("🕸️ Pass Network Analysis")
        if analysis_results.get('key_players'):
            key_players = analysis_results['key_players']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Playmaker", f"Player {key_players.get('playmaker', 'N/A')}")
            with col2:
                st.metric("Connector", f"Player {key_players.get('connector', 'N/A')}")
            with col3:
                st.metric("Most Influential", f"Player {key_players.get('influential', 'N/A')}")
        
        # Passing patterns
        if analysis_results.get('passing_patterns'):
            patterns = analysis_results['passing_patterns']
            st.bar_chart(patterns)
        
        # Commentary playback
        st.subheader("🎙️ Match Commentary")
        if st.button("Generate Match Commentary"):
            with st.spinner("Generating commentary..."):
                for event in filtered_events[:5]:  # First 5 events
                    commentary = self.commentary_engine.generate_commentary(event, language)
                    st.write(f"**{event.get('type', 'Event')}:** {commentary}")
                    
                    # Generate audio
                    audio_buffer = self.commentary_engine.text_to_speech(commentary, language)
                    if audio_buffer:
                        st.audio(audio_buffer)
        
        return analysis_results

# Streamlit app integration
def run_enhanced_dashboard():
    st.set_page_config(
        page_title="Enhanced Football Analysis",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Enhanced Football Analysis System")
    st.markdown("Advanced AI-powered football match analysis with real-time insights")
    
    # Initialize system
    if 'analysis_system' not in st.session_state:
        st.session_state.analysis_system = EnhancedFootballAnalysisSystem()
    
    system = st.session_state.analysis_system
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model optimization options
        use_optimized_models = st.checkbox("Use Optimized Models (ONNX/TensorRT)")
        if use_optimized_models:
            model_paths = st.text_area("Model Paths (one per line)", 
                                     "models/yolo_optimized.onnx\nmodels/ball_detector.onnx")
            if model_paths:
                paths = [p.strip() for p in model_paths.split('\n') if p.strip()]
                system.initialize_optimized_models(paths)
        
        # Analysis options
        use_pose_estimation = st.checkbox("Enable Pose Estimation", value=True)
        language = st.selectbox("Commentary Language", 
                               ['en', 'es', 'fr', 'de', 'it', 'pt', 'ar', 'ja'])
    
    # Main interface
    uploaded_file = st.file_uploader("Upload Match Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file:
        # Save uploaded file
        video_path = f"temp_{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        if st.button("🚀 Start Enhanced Analysis"):
            with st.spinner("Running enhanced analysis..."):
                results = system.process_video_enhanced(
                    video_path, 
                    language=language,
                    use_pose=use_pose_estimation
                )
                
                if results:
                    st.success("✅ Analysis Complete!")
                    
                    # Create interactive dashboard
                    system.create_interactive_dashboard(results)
                    
                    # Download options
                    st.subheader("📥 Download Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Download Processed Video"):
                            with open("outputs/enhanced_analysis.mp4", "rb") as f:
                                st.download_button("📹 Download Video", f, "enhanced_analysis.mp4")
                    
                    with col2:
                        if st.button("Download Analysis Report"):
                            import json
                            report_data = json.dumps(results, indent=2, default=str)
                            st.download_button("📊 Download Report", report_data, "analysis_report.json")
                    
                    with col3:
                        if st.button("Download Heatmaps"):
                            st.info("Heatmap download feature coming soon!")

if __name__ == "__main__":
    run_enhanced_dashboard()