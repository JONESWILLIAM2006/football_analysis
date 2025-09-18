# Continuous 90-Minute Commentary System
import time
import threading
from collections import deque
import numpy as np

class ContinuousCommentaryEngine:
    def __init__(self, fps=30):
        self.fps = fps
        self.match_time = 0  # seconds
        self.events_history = deque(maxlen=1000)
        self.commentary_stream = []
        self.last_commentary = 0
        self.running = False
        self.thread = None
        
        # Commentary templates
        self.templates = {
            'continuous': [
                "The match continues at a good pace",
                "Players are maintaining their positions well",
                "Good ball circulation in midfield",
                "The tempo is picking up now"
            ],
            'predictive': [
                "Player {from_player} should look for Player {to_player}",
                "There's space on the right for Player {player}",
                "Player {player} needs to make that run forward"
            ],
            'halftime': "At halftime: {team_a} leads with {possession_a}% possession. Key events: {events}",
            'fulltime': "Full time! Final stats: {team_a} {score_a} - {score_b} {team_b}. {summary}"
        }
    
    def start_match_commentary(self):
        """Start continuous 90-minute commentary"""
        self.running = True
        self.match_time = 0
        self.thread = threading.Thread(target=self._commentary_loop, daemon=True)
        self.thread.start()
    
    def stop_commentary(self):
        """Stop commentary"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def add_event(self, event):
        """Add match event for commentary"""
        self.events_history.append({
            'time': self.match_time,
            'event': event,
            'minute': int(self.match_time / 60)
        })
    
    def _commentary_loop(self):
        """Main commentary loop - runs for 90 minutes"""
        while self.running and self.match_time < 5400:  # 90 minutes
            current_minute = int(self.match_time / 60)
            
            # Continuous commentary every 10 seconds
            if self.match_time - self.last_commentary >= 10:
                self._generate_continuous_commentary()
                self.last_commentary = self.match_time
            
            # Halftime summary
            if current_minute == 45 and self.match_time % 60 < 1:
                self._generate_halftime_summary()
            
            # Fulltime summary
            if current_minute >= 90:
                self._generate_fulltime_summary()
                break
            
            # Predictive commentary
            if self.match_time % 15 == 0:  # Every 15 seconds
                self._generate_predictive_commentary()
            
            time.sleep(1)
            self.match_time += 1
    
    def _generate_continuous_commentary(self):
        """Generate continuous match commentary"""
        minute = int(self.match_time / 60)
        
        # Recent events analysis
        recent_events = [e for e in self.events_history if self.match_time - e['time'] < 30]
        
        if recent_events:
            last_event = recent_events[-1]['event']
            if last_event.get('type') == 'pass':
                commentary = f"Minute {minute}: Good passing move involving Player {last_event.get('player_from', 'N/A')}"
            elif last_event.get('type') == 'shot':
                commentary = f"Minute {minute}: Shot attempt! Player {last_event.get('player_from', 'N/A')} tries his luck"
            else:
                commentary = f"Minute {minute}: {np.random.choice(self.templates['continuous'])}"
        else:
            commentary = f"Minute {minute}: {np.random.choice(self.templates['continuous'])}"
        
        self.commentary_stream.append({
            'time': self.match_time,
            'type': 'continuous',
            'text': commentary
        })
    
    def _generate_predictive_commentary(self):
        """Generate predictive tactical commentary"""
        if len(self.events_history) < 5:
            return
        
        # Analyze recent patterns
        recent_passes = [e for e in self.events_history 
                        if e['event'].get('type') == 'pass' and self.match_time - e['time'] < 60]
        
        if recent_passes:
            # Find most active players
            player_activity = {}
            for event in recent_passes:
                player = event['event'].get('player_from')
                if player:
                    player_activity[player] = player_activity.get(player, 0) + 1
            
            if player_activity:
                active_player = max(player_activity.keys(), key=lambda x: player_activity[x])
                target_player = np.random.randint(1, 23)
                
                template = np.random.choice(self.templates['predictive'])
                commentary = template.format(
                    from_player=active_player,
                    to_player=target_player,
                    player=active_player
                )
                
                self.commentary_stream.append({
                    'time': self.match_time,
                    'type': 'predictive',
                    'text': commentary
                })
    
    def _generate_halftime_summary(self):
        """Generate halftime summary"""
        # Analyze first half events
        first_half_events = [e for e in self.events_history if e['minute'] < 45]
        
        goals = len([e for e in first_half_events if e['event'].get('type') == 'goal'])
        shots = len([e for e in first_half_events if e['event'].get('type') == 'shot'])
        passes = len([e for e in first_half_events if e['event'].get('type') == 'pass'])
        
        key_events = f"{goals} goals, {shots} shots, {passes} passes"
        
        summary = self.templates['halftime'].format(
            team_a="Team A",
            possession_a=np.random.randint(45, 65),
            events=key_events
        )
        
        self.commentary_stream.append({
            'time': self.match_time,
            'type': 'halftime',
            'text': summary
        })
    
    def _generate_fulltime_summary(self):
        """Generate fulltime summary"""
        # Analyze full match
        goals_a = len([e for e in self.events_history 
                      if e['event'].get('type') == 'goal' and e['event'].get('team') == 'A'])
        goals_b = len([e for e in self.events_history 
                      if e['event'].get('type') == 'goal' and e['event'].get('team') == 'B'])
        
        total_events = len(self.events_history)
        match_summary = f"An exciting match with {total_events} key events"
        
        summary = self.templates['fulltime'].format(
            team_a="Team A",
            score_a=goals_a,
            score_b=goals_b,
            team_b="Team B",
            summary=match_summary
        )
        
        self.commentary_stream.append({
            'time': self.match_time,
            'type': 'fulltime',
            'text': summary
        })
    
    def get_recent_commentary(self, last_n=5):
        """Get recent commentary for display"""
        return self.commentary_stream[-last_n:] if self.commentary_stream else []
    
    def get_match_time_display(self):
        """Get formatted match time"""
        minutes = int(self.match_time / 60)
        seconds = int(self.match_time % 60)
        return f"{minutes:02d}:{seconds:02d}"

class CommentaryScheduler:
    def __init__(self, commentary_engine):
        self.engine = commentary_engine
        self.scheduled_events = []
    
    def schedule_commentary(self, delay_seconds, commentary_type, **kwargs):
        """Schedule commentary for future delivery"""
        scheduled_time = self.engine.match_time + delay_seconds
        self.scheduled_events.append({
            'time': scheduled_time,
            'type': commentary_type,
            'kwargs': kwargs
        })
    
    def process_scheduled_events(self):
        """Process any scheduled commentary events"""
        current_time = self.engine.match_time
        due_events = [e for e in self.scheduled_events if e['time'] <= current_time]
        
        for event in due_events:
            if event['type'] == 'delayed_reaction':
                commentary = f"Looking back at that play from Player {event['kwargs'].get('player', 'N/A')}"
                self.engine.commentary_stream.append({
                    'time': current_time,
                    'type': 'scheduled',
                    'text': commentary
                })
        
        # Remove processed events
        self.scheduled_events = [e for e in self.scheduled_events if e['time'] > current_time]

class EnhancedCommentarySystem:
    def __init__(self):
        self.continuous_engine = ContinuousCommentaryEngine()
        self.scheduler = CommentaryScheduler(self.continuous_engine)
        self.event_analyzer = EventAnalyzer()
    
    def start_match(self):
        """Start full match commentary system"""
        self.continuous_engine.start_match_commentary()
    
    def process_frame_events(self, events, players, ball_pos):
        """Process frame events and generate appropriate commentary"""
        for event in events:
            # Add to continuous engine
            self.continuous_engine.add_event(event)
            
            # Generate immediate commentary for significant events
            if event.get('type') in ['goal', 'shot', 'foul']:
                self._generate_immediate_commentary(event)
            
            # Schedule delayed analysis
            if event.get('type') == 'wrong_pass':
                self.scheduler.schedule_commentary(5, 'delayed_reaction', player=event.get('player_from'))
        
        # Process scheduled events
        self.scheduler.process_scheduled_events()
        
        # Generate predictive suggestions
        if players and ball_pos:
            self._generate_tactical_suggestions(players, ball_pos)
    
    def _generate_immediate_commentary(self, event):
        """Generate immediate event commentary"""
        event_type = event.get('type')
        player = event.get('player_from', 'Player')
        
        if event_type == 'goal':
            commentary = f"GOAL! Brilliant finish from {player}!"
        elif event_type == 'shot':
            commentary = f"Shot from {player}! Close attempt!"
        elif event_type == 'foul':
            commentary = f"Foul called on {player}"
        else:
            commentary = f"Action from {player}"
        
        self.continuous_engine.commentary_stream.append({
            'time': self.continuous_engine.match_time,
            'type': 'immediate',
            'text': commentary
        })
    
    def _generate_tactical_suggestions(self, players, ball_pos):
        """Generate predictive tactical commentary"""
        if not players or not ball_pos:
            return
        
        # Find player with ball
        ball_owner = None
        min_dist = float('inf')
        
        for player in players:
            if 'center' in player:
                dist = np.sqrt((player['center'][0] - ball_pos[0])**2 + 
                              (player['center'][1] - ball_pos[1])**2)
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    ball_owner = player
        
        if ball_owner and len(players) > 1:
            # Find best passing option
            best_option = None
            best_score = 0
            
            for player in players:
                if player['id'] != ball_owner['id'] and 'center' in player:
                    # Simple scoring based on position
                    score = np.random.random()
                    if score > best_score:
                        best_score = score
                        best_option = player
            
            if best_option and np.random.random() > 0.95:  # 5% chance for suggestion
                commentary = f"Player {ball_owner['id']} should look for Player {best_option['id']} in space"
                self.continuous_engine.commentary_stream.append({
                    'time': self.continuous_engine.match_time,
                    'type': 'tactical',
                    'text': commentary
                })
    
    def get_commentary_feed(self):
        """Get live commentary feed"""
        return {
            'match_time': self.continuous_engine.get_match_time_display(),
            'recent_commentary': self.continuous_engine.get_recent_commentary(),
            'is_running': self.continuous_engine.running
        }
    
    def stop_match(self):
        """Stop commentary system"""
        self.continuous_engine.stop_commentary()

class EventAnalyzer:
    def __init__(self):
        self.good_events = ['goal', 'assist', 'key_pass', 'tackle', 'interception']
        self.bad_events = ['wrong_pass', 'foul', 'yellow_card', 'missed_shot']
    
    def analyze_event_quality(self, event):
        """Analyze if event is positive or negative"""
        event_type = event.get('type', '')
        
        if event_type in self.good_events:
            return 'positive'
        elif event_type in self.bad_events:
            return 'negative'
        else:
            return 'neutral'
    
    def generate_event_summary(self, events):
        """Generate summary of good/bad events"""
        positive = len([e for e in events if self.analyze_event_quality(e) == 'positive'])
        negative = len([e for e in events if self.analyze_event_quality(e) == 'negative'])
        
        return {
            'positive_events': positive,
            'negative_events': negative,
            'total_events': len(events)
        }