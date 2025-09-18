# Commentary Integration for Football Analysis
from continuous_commentary import EnhancedCommentarySystem

class CommentaryIntegration:
    def __init__(self):
        self.continuous_commentary = EnhancedCommentarySystem()
        self.match_started = False
    
    def start_match_commentary(self):
        """Start continuous 90-minute commentary"""
        if not self.match_started:
            self.continuous_commentary.start_match()
            self.match_started = True
    
    def process_frame_commentary(self, events, players, ball_pos):
        """Process frame events for commentary"""
        if self.match_started:
            self.continuous_commentary.process_frame_events(events, players, ball_pos)
    
    def get_commentary_display(self):
        """Get commentary for UI display"""
        if self.match_started:
            return self.continuous_commentary.get_commentary_feed()
        return {'match_time': '00:00', 'recent_commentary': [], 'is_running': False}
    
    def stop_commentary(self):
        """Stop commentary system"""
        if self.match_started:
            self.continuous_commentary.stop_match()
            self.match_started = False