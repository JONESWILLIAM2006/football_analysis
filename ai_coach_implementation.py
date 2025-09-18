def render_ai_coach_page():
    """Render the AI coach page"""
    st.header("🤖 AI Coach")
    st.info("Chat with your AI football coach for tactical insights")
    
    if 'coach' not in st.session_state:
        st.session_state.coach = ConversationalAICoach()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Coach Settings")
        coach_style = st.selectbox("Coaching Style", ["Generic", "Pep Guardiola", "Hansi Flick"])
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.coach.clear_memory()
            st.rerun()
    
    with col1:
        st.subheader("💬 Chat with AI Coach")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if prompt := st.chat_input("Ask your coach about tactics, players, or strategy..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Coach is analyzing..."):
                    response = st.session_state.coach.process_query(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.expander("💡 Example Questions"):
        st.write("""
        - "Show me the heatmap for player 7"
        - "Compare passing accuracy of player 7 and 10"
        - "What events happened in the final third?"
        - "Analyze the formation we're using"
        - "Find all wrong passes in the match"
        - "Show tactical insights for this game"
        """)

def render_match_analysis_page():
    """Render the match analysis page"""
    st.header("📊 Match Analysis")
    st.info("Upload a football match video for comprehensive analysis")
    
    uploaded_video = st.file_uploader("Upload Match Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        video_path = f"temp_video_{int(time.time())}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_formation = st.selectbox("Team Formation", ["4-4-2", "4-3-3", "3-5-2"])
            enable_var = st.checkbox("Enable VAR Analysis", value=True)
        with col2:
            generate_report = st.checkbox("Generate PDF Report", value=True)
            save_highlights = st.checkbox("Save Highlights", value=False)
        
        if st.button("🚀 Start Analysis", type="primary"):
            with st.spinner("Analyzing match video..."):
                try:
                    detection_engine = DetectionEngine()
                    predictive_tactician = PredictiveTactician()
                    tactical_analyzer = TacticalAnalyzer()
                    xg_xa_model = xG_xA_Model()
                    
                    events, wrong_passes, foul_events, training_rows = detection_engine.run_detection(
                        video_path, predictive_tactician, tactical_analyzer, xg_xa_model, selected_formation
                    )
                    
                    st.success(f"✅ Analysis complete! Found {len(events)} events")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Events", len(events))
                    with col2:
                        st.metric("Wrong Passes", len(wrong_passes))
                    with col3:
                        st.metric("Potential Fouls", len(foul_events))
                    with col4:
                        passes = [e for e in events if 'pass' in e[1]]
                        st.metric("Total Passes", len(passes))
                    
                    if os.path.exists("outputs/processed_video.mp4"):
                        st.video("outputs/processed_video.mp4")
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                finally:
                    if os.path.exists(video_path):
                        os.remove(video_path)

def render_whatif_page():
    """Render the what-if scenarios page"""
    st.header("🎬 What-If Scenarios")
    st.info("Generate hypothetical tactical scenarios")
    
    st.subheader("🎯 Scenario Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Upload Match Video**")
        uploaded_video = st.file_uploader("Video for What-If Analysis", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video:
            coach_style = st.selectbox("Tactical Philosophy", ["Generic", "Pep Guardiola", "Jurgen Klopp"])
            scenario_type = st.selectbox("Scenario Type", ["Wrong Pass Correction", "Tactical Formation", "Counter Attack"])
    
    with col2:
        st.write("**Animation Settings**")
        animation_speed = st.slider("Animation Speed", 0.5, 2.0, 1.0)
        show_analysis = st.checkbox("Show Tactical Analysis", value=True)
        include_commentary = st.checkbox("Include AI Commentary", value=True)
    
    if uploaded_video and st.button("🎬 Generate What-If Scenario", type="primary"):
        with st.spinner("Generating tactical scenario..."):
            try:
                video_path = f"temp_video_{int(time.time())}.mp4"
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.read())
                
                scenario_generator = TacticalScenarioGenerator()
                animation_engine = AnimationEngine()
                
                detection_engine = DetectionEngine()
                predictive_tactician = PredictiveTactician()
                tactical_analyzer = TacticalAnalyzer()
                xg_xa_model = xG_xA_Model()
                
                events, wrong_passes, foul_events, training_rows = detection_engine.run_detection(
                    video_path, predictive_tactician, tactical_analyzer, xg_xa_model, "4-3-3"
                )
                
                if wrong_passes:
                    st.success(f"Found {len(wrong_passes)} wrong passes to analyze")
                    
                    wrong_pass = wrong_passes[0]
                    optimal_scenario = scenario_generator.generate_optimal_scenario(wrong_pass, coach_style)
                    
                    if optimal_scenario:
                        st.subheader("📊 Tactical Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Optimal Target", f"Player {optimal_scenario['optimal_target']}")
                        with col2:
                            st.metric("Pass Type", optimal_scenario['pass_type'].replace('_', ' ').title())
                        with col3:
                            st.metric("Confidence", f"{optimal_scenario['confidence_score']:.2f}")
                        
                        st.write(f"**Reasoning:** {optimal_scenario['reasoning']}")
                        st.write(f"**Expected Outcome:** {optimal_scenario['expected_outcome']}")
                        
                        animation_path = animation_engine.create_tactical_scenario_animation(
                            wrong_pass, coach_style, "animations/scenario.mp4"
                        )
                        
                        if animation_path and os.path.exists(animation_path):
                            st.subheader("🎬 Tactical Scenario Animation")
                            st.video(animation_path)
                
                else:
                    st.info("No wrong passes detected in this video segment")
                
            except Exception as e:
                st.error(f"Scenario generation failed: {str(e)}")
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)
    
    with st.expander("ℹ️ About What-If Analysis"):
        st.write("""
        **What-If Scenarios** use AI to:
        
        1. **Detect Wrong Passes**: Identify turnovers and poor decisions
        2. **Generate Alternatives**: Find optimal pass targets using tactical philosophy
        3. **Predict Outcomes**: Show how the game could have unfolded differently
        4. **Create Animations**: Visualize the alternative scenario with player movements
        
        **Tactical Philosophies:**
        - **Generic**: Balanced approach focusing on open space
        - **Pep Guardiola**: Short passes, possession retention, positional play
        - **Jurgen Klopp**: Vertical passes, quick transitions, wing play
        """)