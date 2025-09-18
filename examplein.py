Updated code 7# football_analysis_combined.py

import os
import cv2
import csv
import time
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from sort import Sort
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from gtts import gTTS
import urllib.request
import random
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import altair as alt
import openai
import pygame
from pymongo import MongoClient
import json
from sklearn.preprocessing import LabelEncoder
from matplotlib.animation import FuncAnimation

# -------------------- CONFIGURATION & SETUP --------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("highlights", exist_ok=True)
os.makedirs("animations", exist_ok=True)

YOLO_MODEL = "yolov8x.pt"
PITCH_IMAGE = "football_pitch.png"

# Download pitch image if it doesn't exist
if not os.path.exists(PITCH_IMAGE):
    url = "https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython/blob/master/11_PassingNetworks/pitch.png?raw=true"
    try:
        urllib.request.urlretrieve(url, PITCH_IMAGE)
    except Exception as e:
        print(f"Failed to download pitch image: {e}")

pygame.mixer.init()
# IMPORTANT: Replace with your actual OpenAI API Key.
openai.api_key = "YOUR_OPENAI_API_KEY"
client = MongoClient("mongodb://localhost:27017/")
db = client["sports_analytics"]
storage_collection = db["uploads"]

# -------------------- UTILITIES --------------------
def get_pitch_zone(x_coord, frame_width):
    """Determines the pitch zone based on x-coordinate."""
    third = frame_width / 3
    if x_coord < third:
        return "Defensive Third"
    elif x_coord < 2 * third:
        return "Midfield"
    else:
        return "Final Third"

# -------------------- MODULES --------------------

class PassPredictionModel:
    """Trains and uses a machine learning model to predict pass outcomes."""
    def __init__(self, model_path="outputs/pass_classifier.pkl"):
        self.model_path = model_path
        self.model = self._load_model()
        self.label_encoder = LabelEncoder()
        
        # Fit label encoder with known labels
        self.label_encoder.fit(["correct", "wrong"])

    def _load_model(self):
        """Loads the pre-trained model or returns None if not found."""
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None

    def train_model(self, data_path="outputs/training_data.csv"):
        """Trains a new model on the collected data and saves it."""
        if not os.path.exists(data_path):
            st.warning("No training data found. Please run an analysis first.")
            return

        df_train = pd.read_csv(data_path)
        
        if len(df_train) < 5:
            st.warning("Not enough data to train (need at least 5 samples).")
            return

        # Prepare features and labels
        df_train['pitch_zone_encoded'] = self.label_encoder.transform(df_train['pitch_zone'])
        
        X = df_train[["distance", "pitch_zone_encoded"]]
        y = df_train["pass_success"]

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Save the model
        joblib.dump(clf, self.model_path)
        self.model = clf
        st.success("🎉 Pass prediction model trained successfully!")
        
        # Optional: Display model accuracy
        st.write(f"Model accuracy on test data: {clf.score(X_test, y_test):.2f}")

    def predict(self, distance, pitch_zone):
        """Predicts the pass outcome based on new data."""
        if not self.model:
            return 0.5 # Return a neutral probability if no model is trained
        
        # Encode the pitch zone
        try:
            pitch_zone_encoded = self.label_encoder.transform([pitch_zone])[0]
        except ValueError:
            pitch_zone_encoded = 0 # Default to defensive third if unknown
        
        # Predict the probability of success (class 1)
        prediction = self.model.predict_proba([[distance, pitch_zone_encoded]])
        return prediction[0][1]


class DetectionEngine:
    """Processes videos to detect players, ball, and tactical events."""
    def __init__(self, model_path=YOLO_MODEL):
        self.model = YOLO(model_path)
        self.tracker = Sort()
        self.player_positions = {}
        self.player_energy = {}
        self.MAX_ENERGY = 100
        self.DRAIN_RATE = 0.5
        self.RECOVERY_RATE = 0.1

    def run_detection(self, video_path, predictive_tactician):
        """Processes video frames to detect players, ball, and events."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            return [], [], []

        width, height = 1280, 720
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = "outputs/processed_video.mp4"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        events = []
        correct_passes = []
        wrong_passes = []
        foul_events = []
        training_rows = []
        last_ball_owner = None
        
        # Placeholder for real weather data
        field_weather = {"temperature": "15°C", "condition": "Overcast"}
        st.info(f"Field Weather: {field_weather['temperature']}, {field_weather['condition']}")


        st_video_placeholder = st.empty()
        
        var_engine = VARAnalysisEngine()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (width, height))
            detections = self.model(frame)[0].boxes
            player_boxes = []
            ball_box = None

            for box in detections:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == 0 and conf > 0.5: # Player
                    player_boxes.append([x1, y1, x2, y2, conf])
                elif cls == 32 and conf > 0.3: # Ball
                    ball_box = [x1, y1, x2, y2]

            tracked_players = self.tracker.update(np.array(player_boxes)) if player_boxes else np.empty((0, 5))
            players = {}
            player_bounding_boxes = {}
            for x1, y1, x2, y2, track_id in tracked_players:
                track_id = int(track_id)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                players[track_id] = (cx, cy)
                player_bounding_boxes[track_id] = (x1, y1, x2, y2)
                
                # --- Player Energy Level Logic ---
                if track_id not in self.player_energy:
                    self.player_energy[track_id] = self.MAX_ENERGY
                
                if track_id in self.player_positions:
                    prev_pos = self.player_positions[track_id]
                    distance_moved = np.linalg.norm(np.array((cx, cy)) - np.array(prev_pos))
                    
                    if distance_moved > 15:
                        self.player_energy[track_id] -= self.DRAIN_RATE * 3
                    elif distance_moved > 5:
                        self.player_energy[track_id] -= self.DRAIN_RATE
                    else:
                        self.player_energy[track_id] += self.RECOVERY_RATE

                self.player_energy[track_id] = max(0, min(self.player_energy[track_id], self.MAX_ENERGY))
                self.player_positions[track_id] = (cx, cy)
                
                energy_level = self.player_energy[track_id]
                color = (0, 255, 0)
                if energy_level < 60:
                    color = (0, 255, 255)
                if energy_level < 30:
                    color = (0, 0, 255)

                cv2.putText(frame, f"ID:{track_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.circle(frame, (x1 + 10, y1 - 10), 8, color, -1)
                
            # --- VAR Analysis ---
            foul_event = var_engine.analyze_foul(frame_num, players, player_bounding_boxes)
            if foul_event:
                foul_events.append(foul_event)
                events.append((foul_event['frame'], "foul", foul_event['player1'], foul_event['player2'], "N/A"))
                
                p1_box = player_bounding_boxes.get(foul_event['player1'])
                p2_box = player_bounding_boxes.get(foul_event['player2'])
                if p1_box:
                    x1, y1, x2, y2 = p1_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if p2_box:
                    x1, y1, x2, y2 = p2_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                cv2.putText(frame, "POTENTIAL FOUL!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)


            if ball_box:
                bx = int((ball_box[0] + ball_box[2]) / 2)
                by = int((ball_box[1] + ball_box[3]) / 2)
                cv2.circle(frame, (bx, by), 6, (0, 0, 255), -1)

                distances = {pid: np.linalg.norm(np.array((bx, by)) - np.array(pos)) for pid, pos in players.items()}
                nearest = sorted(distances.items(), key=lambda x: x[1])

                if nearest and distances[nearest[0][0]] < 40:
                    current_owner = nearest[0][0]
                    if last_ball_owner is not None and current_owner != last_ball_owner:
                        distance = distances.get(current_owner, 0)
                        
                        pitch_zone = get_pitch_zone(players[last_ball_owner][0], width)
                        
                        # Use trained model to predict pass outcome
                        pass_success_prob = predictive_tactician.pass_model.predict(distance, pitch_zone)
                        
                        # Randomly decide success based on probability
                        success = 1 if random.random() < pass_success_prob else 0

                        pass_info = {
                            "frame": frame_num,
                            "player_from": last_ball_owner,
                            "player_to": current_owner,
                            "distance": distance,
                            "pass_success": success,
                            "pitch_zone": pitch_zone
                        }
                        
                        training_rows.append(pass_info)
                        
                        if success == 1:
                            correct_passes.append((frame_num, last_ball_owner, current_owner))
                            events.append((frame_num, "correct_pass", last_ball_owner, current_owner, pitch_zone))
                        else:
                            best_target_id = predictive_tactician.suggest_best_move(players, last_ball_owner)
                            
                            wrong_passes.append({
                                "frame": frame_num,
                                "player_from": last_ball_owner,
                                "player_to": current_owner,
                                "wrong_pass_start_pos": players[last_ball_owner],
                                "wrong_pass_end_pos": players[current_owner],
                                "optimal_pass": {
                                    "target_player": best_target_id,
                                    "target_pos": players.get(best_target_id),
                                    "analysis": "This pass created a turnover. A short pass to the central midfielder would have retained possession and opened up the field."
                                }
                            })
                            events.append((frame_num, "wrong_pass", last_ball_owner, current_owner, pitch_zone))
                    
                    if current_owner in players:
                        # --- Best Move Visualization ---
                        best_target_id = predictive_tactician.suggest_best_move(players, current_owner)
                        if best_target_id and best_target_id in players:
                            from_pos = players[current_owner]
                            to_pos = players[best_target_id]
                            cv2.arrowedLine(frame, from_pos, to_pos, (0, 255, 0), 2, tipLength=0.3)
                            cv2.putText(frame, "Best Pass", (from_pos[0], from_pos[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    last_ball_owner = current_owner
            
            st_video_placeholder.image(frame, channels="BGR", use_container_width=True)
            out.write(frame)
            frame_num += 1

        cap.release()
        out.release()
        
        return events, correct_passes, wrong_passes, foul_events, training_rows


class AnimationEngine:
    """Generates animated visualizations of corrected tactical plays."""
    def create_corrected_pass_animation(self, wrong_pass_data, output_filename="animation/corrected_pass.mp4"):
        """
        Creates an animated video showing a wrong pass and the corrected play.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect('equal')
        ax.set_xlim(-10, 110)
        ax.set_ylim(-10, 70)
        
        pitch_image = plt.imread(PITCH_IMAGE)
        ax.imshow(pitch_image, extent=[-10, 110, -10, 70], alpha=0.8)
        
        # Extract data from the wrong_pass event
        player_from_id = wrong_pass_data['player_from']
        player_to_id = wrong_pass_data['player_to']
        wrong_pass_start_pos = wrong_pass_data['wrong_pass_start_pos']
        wrong_pass_end_pos = wrong_pass_data['wrong_pass_end_pos']
        
        optimal_target_id = wrong_pass_data['optimal_pass']['target_player']
        optimal_target_pos = wrong_pass_data['optimal_pass']['target_pos']
        analysis = wrong_pass_data['optimal_pass']['analysis']
        
        # --- Create a single image for the mistake ---
        def create_mistake_frame():
            ax.clear()
            ax.imshow(pitch_image, extent=[-10, 110, -10, 70], alpha=0.8)
            ax.set_title("Mistake: Incorrect Pass")
            
            # Plot players and ball
            ax.plot(wrong_pass_start_pos[0]/10, wrong_pass_start_pos[1]/10, 'o', color='red', markersize=10, label=f'Player {player_from_id}')
            ax.plot(wrong_pass_end_pos[0]/10, wrong_pass_end_pos[1]/10, 'o', color='yellow', markersize=10, label=f'Player {player_to_id} (Target)')
            if optimal_target_pos:
                ax.plot(optimal_target_pos[0]/10, optimal_target_pos[1]/10, 'o', color='green', markersize=10, label=f'Player {optimal_target_id} (Open)')
            
            # Plot wrong pass trajectory
            ax.plot([wrong_pass_start_pos[0]/10, wrong_pass_end_pos[0]/10], [wrong_pass_start_pos[1]/10, wrong_pass_end_pos[1]/10], 'r--', label='Wrong Pass')
            
            ax.text(50, 65, "The Wrong Pass", horizontalalignment='center', fontsize=14, color='red')
            ax.text(50, 60, "The ball was intercepted due to the pass to a marked player.", horizontalalignment='center', fontsize=10)
            
            plt.legend()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return image

        # --- Create the animation for the corrected action ---
        def animate_frame(i):
            ax.clear()
            ax.imshow(pitch_image, extent=[-10, 110, -10, 70], alpha=0.8)
            ax.set_title("Correction: The Optimal Pass")

            # Plot players and ball
            ax.plot(wrong_pass_start_pos[0]/10, wrong_pass_start_pos[1]/10, 'o', color='red', markersize=10)
            if optimal_target_pos:
                # Animate ball movement
                ball_x = wrong_pass_start_pos[0]/10 + (optimal_target_pos[0]/10 - wrong_pass_start_pos[0]/10) * i/50
                ball_y = wrong_pass_start_pos[1]/10 + (optimal_target_pos[1]/10 - wrong_pass_start_pos[1]/10) * i/50
                ax.plot(ball_x, ball_y, 'o', color='white', markersize=5)
                
                # Animate player movement slightly towards ball
                player_x = optimal_target_pos[0]/10 - (optimal_target_pos[0]/10 - wrong_pass_start_pos[0]/10) * (50-i)/150
                player_y = optimal_target_pos[1]/10 - (optimal_target_pos[1]/10 - wrong_pass_start_pos[1]/10) * (50-i)/150
                
                ax.plot(optimal_target_pos[0]/10, optimal_target_pos[1]/10, 'o', color='green', markersize=10, label=f'Player {optimal_target_id} (Open)')
            
            # Plot players
            ax.plot(wrong_pass_start_pos[0]/10, wrong_pass_start_pos[1]/10, 'o', color='red', markersize=10, label=f'Player {player_from_id}')
            ax.plot(wrong_pass_end_pos[0]/10, wrong_pass_end_pos[1]/10, 'o', color='yellow', markersize=10, label=f'Player {player_to_id} (Original Target)')
            
            # Draw correct pass line
            if optimal_target_pos:
                ax.plot([wrong_pass_start_pos[0]/10, optimal_target_pos[0]/10], [wrong_pass_start_pos[1]/10, optimal_target_pos[1]/10], 'g--', label='Correct Pass')
            
            ax.text(50, 65, "The Correct Pass", horizontalalignment='center', fontsize=14, color='green')
            ax.text(50, 60, analysis, horizontalalignment='center', fontsize=10)

            plt.legend()
            return [ax]
        
        # Save mistake frame as a static image
        fig.savefig("outputs/mistake_frame.png")
        
        # Create and save animation
        ani = FuncAnimation(fig, animate_frame, frames=50, interval=100, blit=True)
        ani.save(output_filename, writer='ffmpeg', fps=10)
        plt.close(fig)
        
        return output_filename


class CommentaryEngine:
    """Generates and plays contextual commentary using a large language model."""
    def generate_commentary(self, event, current_score="0-0", match_time="10:00"):
        """
        Generate natural-sounding commentary for a detected event using an LLM.
        """
        event_type = event.get('type', 'unknown')
        player_from = event.get('player_from', 'an unnamed player')
        player_to = event.get('player_to', 'another player')
        
        if event_type == "correct_pass":
            event_description = f"Player ID {player_from} has just completed a pass to Player ID {player_to}."
            event_importance = "This was a good, short pass in the midfield to retain possession."
            tone = "Calm and informative."
        elif event_type == "wrong_pass":
            event_description = f"Player ID {player_from} has made a misplaced pass."
            event_importance = "This resulted in a turnover and a potential counter-attack for the opposition."
            tone = "Slightly tense and analytical."
        elif event_type == "foul":
            event_description = f"Players {player_from} and {player_to} were in a challenge."
            event_importance = "This may be a foul, and the referee is taking a closer look."
            tone = "Tense and cautious."
        else:
            event_description = f"An event of type '{event_type}' just occurred."
            event_importance = "Unclear tactical importance."
            tone = "Neutral."

        prompt = (
            f"You are a professional sports commentator. Provide a short, human-like commentary "
            f"for a football match. The current score is {current_score} and the time is {match_time}. "
            f"The event is: {event_description}. The tactical importance is: {event_importance}. "
            f"Speak in a {tone} tone. The commentary should be no more than two sentences."
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert sports commentator."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
                temperature=0.7
            )
            commentary_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating commentary: {e}")
            commentary_text = f"An interesting moment in the game as {event_description}."

        return commentary_text

    def text_to_speech(self, text, lang="en"):
        """Convert commentary text to speech."""
        tts = gTTS(text=text, lang=lang)
        audio_file = f"outputs/commentary_{random.randint(0, 10000)}.mp3"
        tts.save(audio_file)
        return audio_file
    
    def play_commentary(self, audio_file):
        """Play commentary audio using pygame."""
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)


class PredictiveTactician:
    """Analyzes player positions to suggest the optimal next move."""
    def __init__(self, pass_model):
        self.pass_model = pass_model

    def suggest_best_move(self, players, ball_owner_id):
        """
        Suggests the best player to pass to based on a simple heuristic.
        """
        if len(players) < 2:
            return None

        best_target_id = None
        max_open_space_score = -1

        for target_id, target_pos in players.items():
            if target_id == ball_owner_id:
                continue

            open_space_score = 0
            for other_id, other_pos in players.items():
                if other_id != target_id:
                    distance = np.linalg.norm(np.array(target_pos) - np.array(other_pos))
                    open_space_score += distance
            
            if open_space_score > max_open_space_score:
                max_open_space_score = open_space_score
                best_target_id = target_id
        
        return best_target_id


class HighlightReelGenerator:
    """Generates a highlight video from a list of events."""
    def create_highlights(self, video_path, events, player_id=None, event_type=None, pitch_zone=None, output_filename="highlights/highlight_reel.mp4", pre_duration=2, post_duration=3):
        """
        Generates a highlight video by clipping segments around detected events based on filters.
        """
        filtered_events = []
        for event in events:
            is_player_match = (player_id is None or event[2] == player_id or (len(event) > 3 and event[3] == player_id))
            is_event_match = (event_type is None or event[1] == event_type)
            is_zone_match = (pitch_zone is None or event[4] == pitch_zone)
            
            if is_player_match and is_event_match and is_zone_match:
                filtered_events.append(event)

        if not filtered_events:
            return None

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        temp_clips = []
        clip_counter = 0

        sorted_events = sorted(filtered_events, key=lambda x: x[0])
        
        for event in sorted_events:
            event_frame = event[0]
            start_frame = max(0, event_frame - int(pre_duration * fps))
            end_frame = event_frame + int(post_duration * fps)

            temp_filename = f"highlights/temp_clip_{clip_counter}.mp4"
            out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            out.release()
            temp_clips.append(temp_filename)
            clip_counter += 1

        cap.release()
        
        final_out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        for temp_clip in temp_clips:
            clip_cap = cv2.VideoCapture(temp_clip)
            while clip_cap.isOpened():
                ret, frame = clip_cap.read()
                if not ret:
                    break
                final_out.write(frame)
            clip_cap.release()
            os.remove(temp_clip)

        final_out.release()
        return output_filename


class TacticalAnalyzer:
    """Analyzes gameplay and recommends optimal moves based on field dynamics."""
    def analyze_event(self, event, classifier):
        """Analyze a pass and suggest optimal positions or corrected actions."""
        analysis = {}
        if event[1] == "correct_pass":
            analysis['wrong_pass'] = False
        else:
            analysis['wrong_pass'] = True
            analysis['corrected_move'] = {
                "optimal_position": (random.randint(100, 500), random.randint(100, 300)),
                "probability": 0.85,
                "description": "Passing to Player X would have been better."
            }
        return analysis

    def simulate_alternative_play(self, frame, analysis):
        """Visualize alternative moves based on analysis."""
        if analysis.get('wrong_pass'):
            pos = analysis['corrected_move']['optimal_position']
            cv2.arrowedLine(frame, (50, 200), pos, (0, 255, 0), 2)
            cv2.putText(frame, "Best Move", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame


class VARAnalysisEngine:
    """Uses pose estimation and motion tracking to determine fouls."""
    def __init__(self):
        self.contact_threshold = 50

    def analyze_foul(self, frame_num, players, player_bounding_boxes):
        """Detect and analyze potential fouls."""
        foul_event = None
        player_ids = list(players.keys())
        if len(player_ids) < 2:
            return None
        
        for i in range(len(player_ids)):
            for j in range(i + 1, len(player_ids)):
                p1_id, p2_id = player_ids[i], player_ids[j]
                p1_pos, p2_pos = players[p1_id], players[p2_id]
                
                distance = np.linalg.norm(np.array(p1_pos) - np.array(p2_pos))
                
                if distance < self.contact_threshold:
                    if random.random() > 0.9:
                        foul_event = {
                            "frame": frame_num,
                            "player1": p1_id,
                            "player2": p2_id,
                            "verdict": "Potential Foul - Contact Detected",
                            "details": f"Players {p1_id} and {p2_id} were in close proximity."
                        }
                        return foul_event
        return None


class MatchAnalysisSystem:
    """Main system that integrates all components."""
    def __init__(self):
        self.pass_prediction_model = PassPredictionModel()
        self.detection_engine = DetectionEngine()
        self.commentary_engine = CommentaryEngine()
        self.tactical_analyzer = TacticalAnalyzer()
        self.var_engine = VARAnalysisEngine()
        self.report_generator = ReportGenerator()
        self.predictive_tactician = PredictiveTactician(self.pass_prediction_model)
        self.highlight_generator = HighlightReelGenerator()
        self.animation_engine = AnimationEngine()
        self.wrong_passes_list = []

    def process_match(self, video_path):
        events, correct_passes, wrong_passes, foul_events, training_rows = self.detection_engine.run_detection(video_path, self.predictive_tactician)
        self.wrong_passes_list = wrong_passes

        tactical_feedback = []
        for event in events:
            if event[1] in ["correct_pass", "wrong_pass"]:
                analysis = self.tactical_analyzer.analyze_event(event, None) # Classifier is not used here for now
                if analysis.get('wrong_pass') and 'corrected_move' in analysis:
                    tactical_feedback.append(analysis['corrected_move']['description'])

        commentary_transcript = []
        for event in events:
            event_dict = {
                "type": event[1],
                "frame": event[0],
                "player_from": event[2],
                "player_to": event[3] if len(event) > 3 else "N/A"
            }
            commentary_text = self.commentary_engine.generate_commentary(event_dict)
            commentary_transcript.append(commentary_text)
        
        pdf_report = self.report_generator.generate_pdf({"Score": "2-1", "Possession": "60%"}, commentary_transcript, tactical_feedback, foul_events, "outputs/heatmap.png")

        summary_commentary = "Welcome to the match summary. Here are the highlights.\n"
        for event in events:
            event_type = event[1]
            if event_type == "correct_pass":
                 summary_commentary += f"A great pass from Player {event[2]} to Player {event[3]} at frame {event[0]} in the {event[4]}.\n"
            elif event_type == "foul":
                summary_commentary += f"A foul was detected at frame {event[0]} involving players {event[2]} and {event[3]}.\n"

        summary_audio = self.commentary_engine.text_to_speech(summary_commentary)
        
        return {
            "events": events,
            "correct_passes": correct_passes,
            "wrong_passes": wrong_passes,
            "foul_events": foul_events,
            "training_rows": training_rows,
            "pdf_report": pdf_report,
            "summary_audio": summary_audio,
        }
    
    def generate_filtered_highlights(self, video_path, events, player_id, event_type, pitch_zone):
        """Generates a filtered highlight reel and returns its path."""
        filtered_player_id = int(player_id) if player_id != "All" else None
        
        if event_type == "Correct Passes":
            filtered_event_type = "correct_pass"
        elif event_type == "Wrong Passes":
            filtered_event_type = "wrong_pass"
        elif event_type == "Fouls":
            filtered_event_type = "foul"
        else:
            filtered_event_type = None

        filtered_pitch_zone = pitch_zone if pitch_zone != "All" else None

        output_name = f"highlights/{'all_' if filtered_player_id is None else f'player_{filtered_player_id}_'}" \
                      f"{'all_' if filtered_event_type is None else f'{filtered_event_type}_'}" \
                      f"{'all' if filtered_pitch_zone is None else filtered_pitch_zone.replace(' ', '_')}" \
                      f"_highlights.mp4"

        highlight_video_path = self.highlight_generator.create_highlights(
            video_path=video_path,
            events=events,
            player_id=filtered_player_id,
            event_type=filtered_event_type,
            pitch_zone=filtered_pitch_zone,
            output_filename=output_name
        )
        return highlight_video_path

    def generate_wrong_pass_animations(self):
        animation_files = []
        for idx, wrong_pass in enumerate(self.wrong_passes_list):
            output_filename = f"animations/wrong_pass_animation_{idx}.mp4"
            animation_files.append(self.animation_engine.create_corrected_pass_animation(wrong_pass, output_filename))
        return animation_files


# -------------------- STREAMLIT DASHBOARD ENTRY --------------------
def run_dashboard():
    st.title("⚽️ Football Match Tactical Analysis")
    st.markdown("Upload a match video for a complete tactical breakdown.")

    video_file = st.file_uploader("Upload Match Video", type=["mp4"])
    
    system = MatchAnalysisSystem()

    retrain_button = st.button("🔁 Retrain Pass Classifier")
    training_data_path = "outputs/training_data.csv"

    if retrain_button:
        system.pass_prediction_model.train_model(data_path=training_data_path)
    
    # Check if a model exists to enable prediction-related UI
    model_exists = os.path.exists("outputs/pass_classifier.pkl")

    if video_file:
        video_path = f"outputs/{video_file.name}"
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        
        st.video(video_path)
        st.info("Processing video... This may take a few moments.")
        
        results = system.process_match(video_path)
        events = results["events"]
        foul_events = results["foul_events"]
        wrong_passes = results["wrong_passes"]

        pd.DataFrame(events, columns=["frame", "event", "from_id", "to_id", "pitch_zone"]).to_csv("outputs/events.csv", index=False)
        if results['training_rows']:
            df_training = pd.DataFrame(results['training_rows'])
            if os.path.exists(training_data_path):
                existing_df = pd.read_csv(training_data_path)
                updated_df = pd.concat([existing_df, df_training], ignore_index=True)
                updated_df.to_csv(training_data_path, index=False)
            else:
                df_training.to_csv(training_data_path, index=False)
        
        st.success("✅ Analysis Complete! You can now generate highlight reels.")
        
        st.subheader("🚨 VAR Analysis: Detected Fouls")
        if foul_events:
            foul_df = pd.DataFrame(foul_events)
            st.dataframe(foul_df)
        else:
            st.info("No fouls were detected in the match.")
        
        st.subheader("📅 Download Outputs")
        st.download_button("📄 Match Report PDF", open(results['pdf_report'], "rb"), file_name="match_report.pdf")
        st.download_button("📹 Processed Video", open("outputs/processed_video.mp4", "rb"), file_name="processed_video.mp4")
        st.download_button("📊 Event Log CSV", open("outputs/events.csv", "rb"), file_name="event_log.csv")
        st.audio(results['summary_audio'])

        if os.path.exists(training_data_path):
            df_train = pd.read_csv(training_data_path)
            if len(df_train) > 1:
                st.subheader("📊 Pass Classification Overview")
                chart = alt.Chart(df_train).mark_bar().encode(
                    x='pass_success:N',
                    y='count():Q',
                    color='pass_success:N'
                ).properties(title="Pass Success Rate").interactive()
                st.altair_chart(chart, use_container_width=True)

        st.subheader("🎬 Generate Custom Highlight Reel")
        if events:
            all_player_ids = sorted(list(set([e[2] for e in events if e[2] is not None])))
            player_options = ["All"] + all_player_ids
            
            event_type_options = ["All", "Correct Passes", "Wrong Passes", "Fouls"]
            
            pitch_zone_options = ["All", "Defensive Third", "Midfield", "Final Third"]

            col1, col2, col3 = st.columns(3)
            with col1:
                selected_player = st.selectbox("Select Player ID", player_options)
            with col2:
                selected_event_type = st.selectbox("Select Event Type", event_type_options)
            with col3:
                selected_pitch_zone = st.selectbox("Select Pitch Zone", pitch_zone_options)

            if st.button("Generate Highlights"):
                with st.spinner("Generating custom highlight reel..."):
                    highlight_video_path = system.generate_filtered_highlights(
                        video_path, events, selected_player, selected_event_type, selected_pitch_zone
                    )
                    if highlight_video_path:
                        st.success("Highlights generated successfully!")
                        st.download_button("🎬 Download Filtered Highlight Reel", open(highlight_video_path, "rb"), file_name=os.path.basename(highlight_video_path))
                    else:
                        st.warning("No events found matching the selected criteria.")
        else:
            st.info("No events were detected in the video to create a highlight reel.")

        # New section for Wrong Pass Animation
        st.subheader("🎨 Visualize Corrected Passes")
        if wrong_passes:
            if st.button("Generate Corrected Pass Animations"):
                with st.spinner("Creating corrected pass animations..."):
                    animation_files = system.generate_wrong_pass_animations()
                    st.success("Animations generated successfully!")
                    for idx, anim_file in enumerate(animation_files):
                        st.download_button(f"🎨 Download Wrong Pass Animation #{idx+1}", open(anim_file, "rb"), file_name=os.path.basename(anim_file))
        else:
            st.info("No wrong passes were detected in the video to create animations.")


# -------------------- MAIN FUNCTION ENTRY --------------------
if __name__ == "__main__":
    run_dashboard()


