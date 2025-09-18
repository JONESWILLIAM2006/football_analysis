# Enhanced Integration Module
import streamlit as st

def integrate_enhanced_mode():
    """Add enhanced mode to main system"""
    mode = st.sidebar.selectbox("Select Mode", ["🎬 Broadcast", "🚀 Enhanced", "📊 Standard"])
    
    if mode == "🚀 Enhanced":
        try:
            from enhanced_football_system import run_enhanced_dashboard
            run_enhanced_dashboard()
            return True
        except ImportError:
            st.error("Enhanced system not available")
            return False
    
    return False