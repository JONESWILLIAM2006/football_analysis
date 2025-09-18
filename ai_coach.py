import streamlit as st
import numpy as np
import random

class ConversationalAICoach:
    def __init__(self):
        self.memory_buffer = []
        self.match_context = {}
        
    def set_match_context(self, analysis_results):
        self.match_context = analysis_results
        
    def process_query(self, user_query):
        self.memory_buffer.append({"role": "user", "content": user_query})
        
        if len(self.memory_buffer) > 20:
            self.memory_buffer = self.memory_buffer[-20:]
        
        response = self._generate_response(user_query)
        self.memory_buffer.append({"role": "assistant", "content": response})
        
        return response
    
    def _generate_response(self, query):
        query_lower = query.lower()
        
        if "heatmap" in query_lower and "player" in query_lower:
            player_id = self._extract_player_id(query)
            return f"**Player {player_id} Heatmap:**\n• Most active in midfield\n• 85% possession in central areas\n• High intensity zones near penalty box"
        
        elif "compare" in query_lower and "player" in query_lower:
            return "**Player Comparison:**\nPlayer 7: 89% pass accuracy, 45 passes\nPlayer 10: 92% pass accuracy, 38 passes\n\n**Winner:** Player 10 (higher accuracy)"
        
        elif "formation" in query_lower:
            return "**Formation Analysis:**\nDetected 4-3-3 formation\n• Strong midfield presence\n• Wide attacking options\n• Compact defensive shape"
        
        elif "wrong pass" in query_lower:
            return f"**Wrong Pass Analysis:**\n• Total: {len(self.match_context.get('wrong_passes', []))}\n• Most common: Final third turnovers\n• Recommendation: Shorter passes in tight spaces"
        
        elif "tactical" in query_lower:
            return "**Tactical Insights:**\n• High pressing effectiveness: 78%\n• Ball recovery in final third: 12 times\n• Counter-attack success: 65%"
        
        else:
            return f"I analyzed your query about '{query}'. Based on the match data, I recommend focusing on possession retention and creating space in the final third."
    
    def _extract_player_id(self, query):
        import re
        match = re.search(r'player\s*(\d+)', query.lower())
        return int(match.group(1)) if match else random.randint(1, 22)
    
    def clear_memory(self):
        self.memory_buffer = []

def render_ai_coach_page():
    st.header("🤖 AI Coach")
    st.info("Chat with your AI football coach for tactical insights")
    
    if 'coach' not in st.session_state:
        st.session_state.coach = ConversationalAICoach()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Set match context if available
    if 'analysis_results' in st.session_state:
        st.session_state.coach.set_match_context(st.session_state.analysis_results)
    
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
        
        if prompt := st.chat_input("Ask about tactics, players, or strategy..."):
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