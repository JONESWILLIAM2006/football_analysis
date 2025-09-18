# Multilingual Commentary System
import random
import time
from gtts import gTTS
from googletrans import Translator
from io import BytesIO
import pygame

class MultilingualCommentary:
    def __init__(self):
        self.translator = Translator()
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ar': 'Arabic',
            'ja': 'Japanese'
        }
        
        # Language-specific commentary templates
        self.templates = {
            'en': {
                'goal': ["GOAL! What a strike!", "It's in the net!", "Brilliant finish!"],
                'pass': ["{player1} finds {player2}", "Nice pass to {player2}", "Through ball to {player2}"],
                'foul': ["Foul called on {player1}", "The referee stops play", "That's a booking!"],
                'shot': ["Shot! Just wide!", "{player1} takes aim!", "Close attempt!"]
            },
            'es': {
                'goal': ["¡GOOOL! ¡Qué golazo!", "¡Está dentro!", "¡Qué remate!"],
                'pass': ["{player1} encuentra a {player2}", "Buen pase a {player2}", "Pase filtrado a {player2}"],
                'foul': ["Falta de {player1}", "El árbitro para el juego", "¡Esa es tarjeta!"],
                'shot': ["¡Disparo! ¡Por poco!", "¡{player1} dispara!", "¡Intento cercano!"]
            },
            'fr': {
                'goal': ["BUT! Quel tir!", "C'est au fond!", "Finition brillante!"],
                'pass': ["{player1} trouve {player2}", "Belle passe vers {player2}", "Passe en profondeur vers {player2}"],
                'foul': ["Faute sifflée contre {player1}", "L'arbitre arrête le jeu", "C'est un carton!"],
                'shot': ["Tir! Juste à côté!", "{player1} tente sa chance!", "Tentative proche!"]
            }
        }
        
        pygame.mixer.init()
    
    def generate_commentary(self, event, language='en', player_names=None):
        if player_names is None:
            player_names = {i: f"Player {i}" for i in range(1, 23)}
        
        event_type = event.get('type', 'pass')
        player1 = event.get('player_from', 1)
        player2 = event.get('player_to', 2)
        
        # Get template for language
        lang_templates = self.templates.get(language, self.templates['en'])
        event_templates = lang_templates.get(event_type, lang_templates['pass'])
        
        # Select random template
        template = random.choice(event_templates)
        
        # Format with player names
        commentary = template.format(
            player1=player_names.get(player1, f"Player {player1}"),
            player2=player_names.get(player2, f"Player {player2}")
        )
        
        return commentary
    
    def text_to_speech(self, text, language='en', slow=False):
        try:
            tts = gTTS(text=text, lang=language, slow=slow)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    def play_commentary(self, audio_buffer):
        if audio_buffer:
            try:
                pygame.mixer.music.load(audio_buffer)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Audio playback error: {e}")
    
    def translate_text(self, text, target_language='es'):
        try:
            translated = self.translator.translate(text, dest=target_language)
            return translated.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def get_synchronized_commentary(self, events, language='en', sync_timestamps=None):
        """Generate synchronized commentary for multiple events"""
        commentary_sequence = []
        
        for i, event in enumerate(events):
            commentary_text = self.generate_commentary(event, language)
            audio_buffer = self.text_to_speech(commentary_text, language)
            
            timestamp = sync_timestamps[i] if sync_timestamps else i * 3.0
            
            commentary_sequence.append({
                'timestamp': timestamp,
                'text': commentary_text,
                'audio': audio_buffer,
                'event': event
            })
        
        return commentary_sequence

class RealTimeHeatmapGenerator:
    def __init__(self, pitch_width=105, pitch_height=68):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.heatmap_data = {}
        self.grid_size = 5  # 5x5 meter grid
        
    def update_heatmap(self, player_id, position):
        if player_id not in self.heatmap_data:
            self.heatmap_data[player_id] = {}
        
        # Convert position to grid coordinates
        grid_x = int(position[0] // self.grid_size)
        grid_y = int(position[1] // self.grid_size)
        grid_key = (grid_x, grid_y)
        
        if grid_key not in self.heatmap_data[player_id]:
            self.heatmap_data[player_id][grid_key] = 0
        
        self.heatmap_data[player_id][grid_key] += 1
    
    def generate_heatmap_image(self, player_id, width=400, height=300):
        import numpy as np
        import cv2
        
        if player_id not in self.heatmap_data:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create heatmap array
        grid_width = self.pitch_width // self.grid_size
        grid_height = self.pitch_height // self.grid_size
        heatmap = np.zeros((grid_height, grid_width))
        
        player_data = self.heatmap_data[player_id]
        max_value = max(player_data.values()) if player_data else 1
        
        for (grid_x, grid_y), count in player_data.items():
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                heatmap[grid_y, grid_x] = count / max_value
        
        # Resize and apply colormap
        heatmap_resized = cv2.resize(heatmap, (width, height))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        return heatmap_colored
    
    def get_player_stats(self, player_id):
        if player_id not in self.heatmap_data:
            return {}
        
        player_data = self.heatmap_data[player_id]
        total_positions = sum(player_data.values())
        
        # Calculate coverage area
        unique_positions = len(player_data)
        coverage_percentage = (unique_positions / (21 * 14)) * 100  # Grid cells
        
        # Find most active zone
        if player_data:
            most_active_zone = max(player_data, key=player_data.get)
            most_active_count = player_data[most_active_zone]
        else:
            most_active_zone = (0, 0)
            most_active_count = 0
        
        return {
            'total_positions': total_positions,
            'unique_zones': unique_positions,
            'coverage_percentage': coverage_percentage,
            'most_active_zone': most_active_zone,
            'most_active_count': most_active_count
        }

class InteractiveControls:
    def __init__(self):
        self.filters = {
            'event_types': [],
            'players': [],
            'time_range': (0, float('inf')),
            'zones': []
        }
    
    def set_event_filter(self, event_types):
        self.filters['event_types'] = event_types
    
    def set_player_filter(self, player_ids):
        self.filters['players'] = player_ids
    
    def set_time_filter(self, start_time, end_time):
        self.filters['time_range'] = (start_time, end_time)
    
    def set_zone_filter(self, zones):
        self.filters['zones'] = zones
    
    def filter_events(self, events):
        filtered = []
        
        for event in events:
            # Check event type filter
            if self.filters['event_types'] and event.get('type') not in self.filters['event_types']:
                continue
            
            # Check player filter
            if self.filters['players']:
                player_from = event.get('player_from')
                player_to = event.get('player_to')
                if player_from not in self.filters['players'] and player_to not in self.filters['players']:
                    continue
            
            # Check time filter
            event_time = event.get('timestamp', 0)
            if not (self.filters['time_range'][0] <= event_time <= self.filters['time_range'][1]):
                continue
            
            # Check zone filter
            if self.filters['zones']:
                event_zone = event.get('zone')
                if event_zone not in self.filters['zones']:
                    continue
            
            filtered.append(event)
        
        return filtered
    
    def get_timeline_data(self, events, bin_size=30):
        """Generate timeline data for visualization"""
        if not events:
            return []
        
        # Group events by time bins
        timeline_data = {}
        
        for event in events:
            timestamp = event.get('timestamp', 0)
            bin_key = int(timestamp // bin_size) * bin_size
            
            if bin_key not in timeline_data:
                timeline_data[bin_key] = {
                    'timestamp': bin_key,
                    'events': [],
                    'count': 0
                }
            
            timeline_data[bin_key]['events'].append(event)
            timeline_data[bin_key]['count'] += 1
        
        return sorted(timeline_data.values(), key=lambda x: x['timestamp'])

class CustomReportGenerator:
    def __init__(self):
        self.chart_types = ['bar', 'line', 'pie', 'heatmap', 'scatter']
    
    def generate_custom_report(self, data, report_config):
        """Generate customized report based on configuration"""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        report_sections = []
        
        for section in report_config.get('sections', []):
            section_type = section.get('type')
            section_data = data.get(section.get('data_key'))
            
            if section_type == 'player_stats':
                chart = self._create_player_stats_chart(section_data, section.get('chart_type', 'bar'))
                report_sections.append({
                    'title': section.get('title', 'Player Statistics'),
                    'chart': chart,
                    'type': 'chart'
                })
            
            elif section_type == 'event_timeline':
                chart = self._create_timeline_chart(section_data)
                report_sections.append({
                    'title': section.get('title', 'Event Timeline'),
                    'chart': chart,
                    'type': 'chart'
                })
            
            elif section_type == 'heatmap':
                heatmap = self._create_heatmap(section_data)
                report_sections.append({
                    'title': section.get('title', 'Player Heatmap'),
                    'chart': heatmap,
                    'type': 'heatmap'
                })
        
        return report_sections
    
    def _create_player_stats_chart(self, stats_data, chart_type='bar'):
        import matplotlib.pyplot as plt
        
        if not stats_data:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        players = list(stats_data.keys())
        values = [stats_data[p].get('passes', 0) for p in players]
        
        if chart_type == 'bar':
            ax.bar(players, values)
        elif chart_type == 'line':
            ax.plot(players, values, marker='o')
        
        ax.set_title('Player Pass Statistics')
        ax.set_xlabel('Players')
        ax.set_ylabel('Number of Passes')
        
        return fig
    
    def _create_timeline_chart(self, timeline_data):
        import matplotlib.pyplot as plt
        
        if not timeline_data:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        timestamps = [item['timestamp'] for item in timeline_data]
        counts = [item['count'] for item in timeline_data]
        
        ax.plot(timestamps, counts, marker='o')
        ax.set_title('Event Timeline')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Number of Events')
        
        return fig
    
    def _create_heatmap(self, heatmap_data):
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not heatmap_data:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert heatmap data to 2D array
        heatmap_array = np.array(heatmap_data)
        
        im = ax.imshow(heatmap_array, cmap='hot', interpolation='nearest')
        ax.set_title('Player Movement Heatmap')
        
        plt.colorbar(im)
        return fig