import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from pymongo import MongoClient
from typing import List, Dict, Any

class EnhancedUserDashboard:
    def __init__(self, mongo_client):
        self.db = mongo_client.football_analysis
        self.jobs_collection = self.db.jobs
        self.users_collection = self.db.users
        
    def render_dashboard(self):
        st.title("🏆 Football Analysis Dashboard")
        
        # Sidebar for navigation
        with st.sidebar:
            st.header("Dashboard Navigation")
            page = st.selectbox("Select View", [
                "📊 Overview", 
                "🔄 Multi-Match Comparison", 
                "👤 Player Profiles",
                "📈 Performance Trends"
            ])
        
        if page == "📊 Overview":
            self._render_overview()
        elif page == "🔄 Multi-Match Comparison":
            self._render_multi_match_comparison()
        elif page == "👤 Player Profiles":
            self._render_player_profiles()
        elif page == "📈 Performance Trends":
            self._render_performance_trends()
    
    def _render_overview(self):
        user_id = st.session_state.get('user_id', 'default_user')
        
        # Get user's analysis history
        analyses = list(self.jobs_collection.find({
            "user_id": user_id,
            "status": "completed"
        }).sort("created_at", -1).limit(10))
        
        if not analyses:
            st.info("No completed analyses found. Upload a video to get started!")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyses", len(analyses))
        with col2:
            avg_duration = np.mean([a.get('processing_time', 0) for a in analyses])
            st.metric("Avg Processing Time", f"{avg_duration:.1f}s")
        with col3:
            total_players = sum([len(a.get('player_stats', {})) for a in analyses])
            st.metric("Players Analyzed", total_players)
        with col4:
            total_events = sum([len(a.get('events', [])) for a in analyses])
            st.metric("Events Detected", total_events)
        
        # Recent analyses table
        st.subheader("📋 Recent Analyses")
        df_analyses = pd.DataFrame([{
            'Date': a['created_at'].strftime('%Y-%m-%d %H:%M'),
            'Match': a.get('match_name', f"Analysis {a['job_id'][:8]}"),
            'Duration': f"{a.get('video_duration', 0):.1f}s",
            'Players': len(a.get('player_stats', {})),
            'Events': len(a.get('events', [])),
            'Job ID': a['job_id']
        } for a in analyses])
        
        st.dataframe(df_analyses, use_container_width=True)
    
    def _render_multi_match_comparison(self):
        st.subheader("🔄 Multi-Match Comparison")
        
        user_id = st.session_state.get('user_id', 'default_user')
        analyses = list(self.jobs_collection.find({
            "user_id": user_id,
            "status": "completed"
        }).sort("created_at", -1))
        
        if len(analyses) < 2:
            st.warning("Need at least 2 completed analyses for comparison.")
            return
        
        # Match selection
        match_options = {f"{a.get('match_name', a['job_id'][:8])} ({a['created_at'].strftime('%Y-%m-%d')})": a['job_id'] 
                        for a in analyses}
        
        selected_matches = st.multiselect(
            "Select matches to compare (2-4 matches):",
            options=list(match_options.keys()),
            max_selections=4
        )
        
        if len(selected_matches) < 2:
            st.info("Please select at least 2 matches to compare.")
            return
        
        # Get selected analysis data
        selected_analyses = [
            next(a for a in analyses if a['job_id'] == match_options[match])
            for match in selected_matches
        ]
        
        # Comparison tabs
        tab1, tab2, tab3 = st.tabs(["📊 Team Metrics", "🗺️ Heatmap Comparison", "⚽ Player Stats"])
        
        with tab1:
            self._render_team_metrics_comparison(selected_analyses, selected_matches)
        
        with tab2:
            self._render_heatmap_comparison(selected_analyses, selected_matches)
        
        with tab3:
            self._render_player_stats_comparison(selected_analyses, selected_matches)
    
    def _render_team_metrics_comparison(self, analyses: List[Dict], match_names: List[str]):
        # Extract team metrics
        metrics_data = []
        for i, analysis in enumerate(analyses):
            match_stats = analysis.get('match_stats', {})
            metrics_data.append({
                'Match': match_names[i],
                'PPDA': match_stats.get('ppda', 0),
                'Possession %': match_stats.get('possession_percentage', 0),
                'Pass Accuracy': match_stats.get('pass_accuracy', 0),
                'Shots': match_stats.get('total_shots', 0),
                'Defensive Actions': match_stats.get('defensive_actions', 0)
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create comparison charts
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['PPDA', 'Possession %', 'Pass Accuracy', 'Shots', 'Defensive Actions', 'Overall Comparison'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "radar"}]]
        )
        
        metrics = ['PPDA', 'Possession %', 'Pass Accuracy', 'Shots', 'Defensive Actions']
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        
        for i, metric in enumerate(metrics):
            row, col = positions[i]
            fig.add_trace(
                go.Bar(x=df_metrics['Match'], y=df_metrics[metric], name=metric, showlegend=False),
                row=row, col=col
            )
        
        # Radar chart for overall comparison
        for i, match in enumerate(match_names):
            match_data = df_metrics[df_metrics['Match'] == match].iloc[0]
            fig.add_trace(
                go.Scatterpolar(
                    r=[match_data[m] for m in metrics],
                    theta=metrics,
                    fill='toself',
                    name=match,
                    showlegend=True
                ),
                row=2, col=3
            )
        
        fig.update_layout(height=800, title_text="Team Metrics Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_heatmap_comparison(self, analyses: List[Dict], match_names: List[str]):
        st.write("### Side-by-Side Heatmap Comparison")
        
        # Player selection for heatmap comparison
        all_players = set()
        for analysis in analyses:
            all_players.update(analysis.get('player_stats', {}).keys())
        
        if not all_players:
            st.warning("No player data available for heatmap comparison.")
            return
        
        selected_player = st.selectbox("Select player for heatmap comparison:", sorted(all_players))
        
        # Create side-by-side heatmaps
        cols = st.columns(len(analyses))
        for i, (analysis, match_name) in enumerate(zip(analyses, match_names)):
            with cols[i]:
                st.write(f"**{match_name}**")
                player_data = analysis.get('player_stats', {}).get(selected_player, {})
                positions = player_data.get('positions', [])
                
                if positions:
                    # Create heatmap visualization
                    fig = go.Figure(data=go.Heatmap(
                        z=self._generate_heatmap_data(positions),
                        colorscale='Reds'
                    ))
                    fig.update_layout(
                        title=f"Player {selected_player}",
                        width=300, height=200,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No position data available")
    
    def _render_player_stats_comparison(self, analyses: List[Dict], match_names: List[str]):
        # Get all players across matches
        all_players = set()
        for analysis in analyses:
            all_players.update(analysis.get('player_stats', {}).keys())
        
        if not all_players:
            st.warning("No player statistics available.")
            return
        
        selected_player = st.selectbox("Select player for detailed comparison:", sorted(all_players))
        
        # Extract player stats across matches
        player_comparison = []
        for i, analysis in enumerate(analyses):
            player_stats = analysis.get('player_stats', {}).get(selected_player, {})
            player_comparison.append({
                'Match': match_names[i],
                'Distance Covered (m)': player_stats.get('distance_covered', 0),
                'Max Speed (km/h)': player_stats.get('max_speed', 0),
                'Avg Speed (km/h)': player_stats.get('avg_speed', 0),
                'Passes': player_stats.get('passes', 0),
                'Pass Accuracy %': player_stats.get('pass_accuracy', 0),
                'Touches': player_stats.get('touches', 0)
            })
        
        df_player = pd.DataFrame(player_comparison)
        
        # Create comparison visualization
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Distance Covered', 'Max Speed', 'Avg Speed', 'Passes', 'Pass Accuracy', 'Touches']
        )
        
        metrics = ['Distance Covered (m)', 'Max Speed (km/h)', 'Avg Speed (km/h)', 'Passes', 'Pass Accuracy %', 'Touches']
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        
        for i, metric in enumerate(metrics):
            row, col = positions[i]
            fig.add_trace(
                go.Bar(x=df_player['Match'], y=df_player[metric], name=metric, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text=f"Player {selected_player} - Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed stats table
        st.subheader("📊 Detailed Statistics")
        st.dataframe(df_player, use_container_width=True)
    
    def _render_player_profiles(self):
        st.subheader("👤 Player Profiles")
        
        user_id = st.session_state.get('user_id', 'default_user')
        analyses = list(self.jobs_collection.find({
            "user_id": user_id,
            "status": "completed"
        }))
        
        # Get all unique players
        all_players = set()
        for analysis in analyses:
            all_players.update(analysis.get('player_stats', {}).keys())
        
        if not all_players:
            st.info("No player data available. Analyze some matches first!")
            return
        
        selected_player = st.selectbox("Select Player:", sorted(all_players))
        
        # Aggregate player data across all matches
        player_matches = []
        for analysis in analyses:
            if selected_player in analysis.get('player_stats', {}):
                player_stats = analysis['player_stats'][selected_player]
                player_matches.append({
                    'date': analysis['created_at'],
                    'match': analysis.get('match_name', analysis['job_id'][:8]),
                    'stats': player_stats
                })
        
        if not player_matches:
            st.warning(f"No data found for player {selected_player}")
            return
        
        # Player overview
        col1, col2, col3, col4 = st.columns(4)
        
        total_distance = sum([m['stats'].get('distance_covered', 0) for m in player_matches])
        avg_speed = np.mean([m['stats'].get('avg_speed', 0) for m in player_matches])
        max_speed = max([m['stats'].get('max_speed', 0) for m in player_matches])
        total_passes = sum([m['stats'].get('passes', 0) for m in player_matches])
        
        with col1:
            st.metric("Total Distance", f"{total_distance:.1f}m")
        with col2:
            st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
        with col3:
            st.metric("Max Speed", f"{max_speed:.1f} km/h")
        with col4:
            st.metric("Total Passes", total_passes)
        
        # Performance trends
        st.subheader("📈 Performance Trends")
        
        df_trends = pd.DataFrame([{
            'Date': m['date'],
            'Match': m['match'],
            'Distance': m['stats'].get('distance_covered', 0),
            'Avg Speed': m['stats'].get('avg_speed', 0),
            'Passes': m['stats'].get('passes', 0),
            'Pass Accuracy': m['stats'].get('pass_accuracy', 0)
        } for m in player_matches])
        
        # Create trend charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Distance Covered', 'Average Speed', 'Passes', 'Pass Accuracy']
        )
        
        fig.add_trace(go.Scatter(x=df_trends['Date'], y=df_trends['Distance'], mode='lines+markers', name='Distance'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_trends['Date'], y=df_trends['Avg Speed'], mode='lines+markers', name='Avg Speed'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df_trends['Date'], y=df_trends['Passes'], mode='lines+markers', name='Passes'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_trends['Date'], y=df_trends['Pass Accuracy'], mode='lines+markers', name='Pass Accuracy'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_trends(self):
        st.subheader("📈 Performance Trends")
        
        user_id = st.session_state.get('user_id', 'default_user')
        analyses = list(self.jobs_collection.find({
            "user_id": user_id,
            "status": "completed"
        }).sort("created_at", 1))
        
        if len(analyses) < 3:
            st.warning("Need at least 3 analyses to show meaningful trends.")
            return
        
        # Team performance trends
        trend_data = []
        for analysis in analyses:
            match_stats = analysis.get('match_stats', {})
            trend_data.append({
                'Date': analysis['created_at'],
                'Match': analysis.get('match_name', analysis['job_id'][:8]),
                'PPDA': match_stats.get('ppda', 0),
                'Possession': match_stats.get('possession_percentage', 0),
                'Pass Accuracy': match_stats.get('pass_accuracy', 0),
                'Shots': match_stats.get('total_shots', 0)
            })
        
        df_trends = pd.DataFrame(trend_data)
        
        # Create trend visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['PPDA Trend', 'Possession Trend', 'Pass Accuracy Trend', 'Shots Trend']
        )
        
        fig.add_trace(go.Scatter(x=df_trends['Date'], y=df_trends['PPDA'], mode='lines+markers', name='PPDA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_trends['Date'], y=df_trends['Possession'], mode='lines+markers', name='Possession'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df_trends['Date'], y=df_trends['Pass Accuracy'], mode='lines+markers', name='Pass Accuracy'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_trends['Date'], y=df_trends['Shots'], mode='lines+markers', name='Shots'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="Team Performance Trends Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance insights
        st.subheader("🔍 Performance Insights")
        
        # Calculate trends
        recent_ppda = df_trends['PPDA'].tail(3).mean()
        early_ppda = df_trends['PPDA'].head(3).mean()
        ppda_trend = "improving" if recent_ppda > early_ppda else "declining"
        
        recent_possession = df_trends['Possession'].tail(3).mean()
        early_possession = df_trends['Possession'].head(3).mean()
        possession_trend = "increasing" if recent_possession > early_possession else "decreasing"
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🎯 **PPDA Trend**: {ppda_trend.title()}")
            st.info(f"⚽ **Possession Trend**: {possession_trend.title()}")
        
        with col2:
            best_match = df_trends.loc[df_trends['Pass Accuracy'].idxmax()]
            st.success(f"🏆 **Best Performance**: {best_match['Match']} ({best_match['Pass Accuracy']:.1f}% pass accuracy)")
    
    def _generate_heatmap_data(self, positions: List[tuple], grid_size: int = 20) -> np.ndarray:
        """Generate heatmap data from player positions"""
        if not positions:
            return np.zeros((grid_size, grid_size))
        
        # Normalize positions to grid
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Create 2D histogram
        heatmap, _, _ = np.histogram2d(x_coords, y_coords, bins=grid_size)
        return heatmap.T