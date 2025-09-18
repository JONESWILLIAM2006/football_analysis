"""
Live & Summary Commentary System for Football Analysis
- LiveCommentator: reacts to incoming events in real-time (90 min commentary)
- SummaryCommentator: generates halftime/fulltime summaries (best/worst events + tactical points)
Audio backends: preferred pyttsx3 (offline). Fallback gTTS -> MP3.
"""

import time
import threading
import heapq
import math
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional

# TTS/backends:
try:
    import pyttsx3
    TTS_BACKEND = "pyttsx3"
except Exception:
    TTS_BACKEND = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

# For playback fallback (local): pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# Utilities
def clamp(v, a, b): return max(a, min(b, v))

# -----------------------------
# COMMENTARY CONTENT / TEMPLATES
# -----------------------------
LIVE_TEMPLATES = {
    "goal": [
        "GOAL!!! What a strike from {player}! That's given the crowd something to roar about.",
        "{player} finds the back of the net — sensational finish at {minute}'!"
    ],
    "shot_on_target": [
        "SHOT ON TARGET from {player}! The keeper is scrambling.",
        "{player} forces the goalie into a save — that was close!"
    ],
    "shot_off_target": [
        "Nearly! {player} with the attempt but not on target.",
    ],
    "pass": [
        "{player_from} plays to {player_to}. Smart distribution.",
        "{player_from} finds {player_to} — good link-up."
    ],
    "wrong_pass": [
        "Oh no — that's a misplaced pass by {player_from} and the opposition pounces.",
        "{player_from} gives it away, poor decision there."
    ],
    "tackle": [
        "Strong tackle by {player} — great defensive work.",
    ],
    "foul": [
        "That's a foul on {player}. The referee has his whistle out.",
    ],
    "interception": [
        "Interception! Good reading of the play.",
    ],
    "counter_attack": [
        "Fast counter from {team}! This could be dangerous.",
    ],
    "default": [
        "{text}"
    ]
}

SUMMARY_TEMPLATES = {
    "best_events_intro": [
        "Now for the key moments that swung the match:",
        "Let's recap the brightest moments of the game."
    ],
    "worst_events_intro": [
        "And now the moments that hurt the team the most:",
    ],
    "tactical_summary_intro": [
        "Tactical takeaways:",
    ],
    "closing": [
        "That's the summary. A great game to dissect; lots to work on in training!"
    ]
}

# ---------------
# SCORING ENGINE
# ---------------
def event_importance_score(event: Dict[str, Any]) -> float:
    """
    Return importance score for an event. 0..10 scale (approx).
    Higher = comment immediately, lower = aggregated.
    Uses keys commonly in your system: type, xg, xg_delta, packing, confidence, is_goal, shot_on_target, severity
    """
    typ = event.get("type", event.get("event", "")).lower()
    score = 0.0

    # Goals are highest
    if typ in ("goal",):
        score += 10.0
        score += event.get("xg_delta", 0) * 5.0
        return score

    # Shots: weight xG and on-target
    if typ == "shot":
        score += 3.0
        if event.get("shot_on_target"):
            score += 2.0
        score += clamp(float(event.get("xg", 0.0)) * 8.0, 0, 4)
        return score

    # Wrong passes that lead to shot/attack
    if typ in ("wrong_pass", "turnover"):
        score += 2.0
        if event.get("leads_to_shot"):
            score += 3.0
        if event.get("packing", 0) > 0:
            score += clamp(event["packing"], 0, 3)
        return score

    # Tactical events
    if typ in ("pressing_trap_success",):
        score += 4.0
        return score

    if typ in ("line_breaking_pass", "packing",):
        score += 3.0
        score += clamp(event.get("packing", 0), 0, 3)
        return score

    # Fouls, cards = moderate-high
    if typ in ("foul", "red_card", "penalty"):
        score += 4.0
        if typ == "red_card":
            score += 4.0
        return score

    # Default (passes, dribbles)
    if typ in ("pass", "correct_pass"):
        # Important if long/through ball/high confidence or progressive
        score += 0.5
        if event.get("progressive", False):
            score += 1.0
        if event.get("long", False):
            score += 1.0
        return score

    return 0.5

# -----------------------------
# AUDIO OUTPUT / PLAYBACK
# -----------------------------
class TTSManager:
    def __init__(self, voice_rate=170, voice_volume=1.0):
        self.backend = TTS_BACKEND
        self.engine = None
        if self.backend == "pyttsx3":
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', voice_rate)
                self.engine.setProperty('volume', voice_volume)
            except Exception:
                self.engine = None
                self.backend = None

    def say_text(self, text: str, sync: bool = True) -> Optional[bytes]:
        """
        Speak the text. If pyttsx3 available, speak directly (low latency).
        Otherwise, fallback to gTTS -> temp mp3 file and play with pygame if available.
        Returns: audio bytes if available (gTTS) otherwise None
        """
        if not text:
            return None

        if self.engine:
            # synchronous speaking
            if sync:
                self.engine.say(text)
                self.engine.runAndWait()
                return None
            else:
                # asynchronous speak on thread
                threading.Thread(target=lambda: (self.engine.say(text), self.engine.runAndWait()), daemon=True).start()
                return None

        # fallback to gTTS (requires internet and can be slower)
        if gTTS is None:
            print("No TTS backend available.")
            return None

        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tmp_path = f"/tmp/commentary_{int(time.time()*1000)}.mp3"
            tts.save(tmp_path)
            # play with pygame if available
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.init()
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print("Playback error:", e)
            # Return bytes (optional)
            with open(tmp_path, "rb") as f:
                data = f.read()
            return data
        except Exception as e:
            print("gTTS error:", e)
            return None

# -----------------------------
# LIVE COMMENTATOR
# -----------------------------
class LiveCommentator:
    def __init__(self,
                 tts_manager: TTSManager,
                 min_gap_seconds=4.0,
                 low_importance_bucket_interval=30.0):
        """
        min_gap_seconds - min seconds between spoken lines (to avoid overlap)
        low_importance_bucket_interval - how often to speak aggregated low-importance chatter
        """
        self.tts = tts_manager
        self.min_gap = min_gap_seconds
        self.bucket_interval = low_importance_bucket_interval

        self.last_speech_time = 0.0
        self.low_priority_buffer = []  # list of (timestamp, event, score)
        self.high_priority_queue = []  # heap of (-score, timestamp, event)
        self.lock = threading.Lock()

        # For summary collection
        self.all_events = []
        self.best_events_heap = []  # min-heap size K storing top events by importance
        self.worst_events_heap = []  # min-heap for worst events (turnovers that cost)
        self.top_k = 6

        # Thread to periodically flush low-priority buffer
        self.running = True
        self._start_background_flusher()

    def _start_background_flusher(self):
        def flusher():
            while self.running:
                time.sleep(1.0)
                now = time.time()
                with self.lock:
                    # If there is a high-priority event and allowed by min_gap => speak it
                    if self.high_priority_queue and now - self.last_speech_time >= self.min_gap:
                        _, ts, ev = heapq.heappop(self.high_priority_queue)
                        self._speak_event(ev)
                        self.last_speech_time = time.time()
                        continue
                    # Else flush low-priority bucket every bucket_interval
                    if now - self.last_speech_time >= self.bucket_interval and self.low_priority_buffer:
                        summary_text = self._aggregate_low_priority()
                        if summary_text:
                            self.tts.say_text(summary_text, sync=True)
                            self.last_speech_time = time.time()
        t = threading.Thread(target=flusher, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    def ingest_event(self, event: Dict[str, Any]):
        """
        Call this for every event as it occurs.
        Event must be a dict with at least 'type' and optionally scoring attributes.
        """
        event = dict(event)  # copy
        event['timestamp'] = time.time()
        score = event_importance_score(event)
        event['_importance_score'] = score

        # Save to all_events for post-match summary
        self.all_events.append(event)
        # Maintain best/worst heaps
        self._update_best_worst(event)

        with self.lock:
            if score >= 6.0:
                # High priority -> push to queue (max heap via negative score)
                heapq.heappush(self.high_priority_queue, (-score, event['timestamp'], event))
            elif score >= 2.0:
                # Medium priority -> attempt to speak when possible (push to queue)
                heapq.heappush(self.high_priority_queue, (-score, event['timestamp'], event))
            else:
                # Low priority -> buffer for aggregation
                self.low_priority_buffer.append((event['timestamp'], event, score))

            # If no backlog and allowed by min_gap, speak immediately high-priority
            now = time.time()
            if self.high_priority_queue and now - self.last_speech_time >= self.min_gap:
                _, ts, ev = heapq.heappop(self.high_priority_queue)
                self._speak_event(ev)
                self.last_speech_time = time.time()

    def _speak_event(self, event: Dict[str, Any]):
        """Convert event to natural language using templates then send to TTS backend"""
        typ = event.get("type", "").lower()
        minute = self._frame_to_minute(event.get("frame")) if event.get("frame") is not None else None

        text = None
        if typ in LIVE_TEMPLATES:
            tmpl = LIVE_TEMPLATES[typ][int(time.time()) % len(LIVE_TEMPLATES[typ])]
            # Fill placeholders defensively
            text = tmpl.format(
                player=event.get("player_name", f"Player {event.get('player_from', '?')}"),
                player_from=event.get("player_from"),
                player_to=event.get("player_to"),
                minute=f"{minute}'" if minute is not None else "the match"
            )
        else:
            text = event.get("text") or f"Event: {typ} occurred."

        # Optionally add extra context for high-impact events
        if event.get("xg_delta"):
            delta = event["xg_delta"]
            text += f" xG change: {delta:+.2f}."

        # Send to TTS (sync speaking ensures no overlap; you can choose async)
        self.tts.say_text(text, sync=True)
        # update last speech time handled by caller

    def _aggregate_low_priority(self) -> Optional[str]:
        """
        Build a brief aggregated line for low priority events (possession, passing rhythm)
        Example: "In the last half-minute, Team A kept possession with a lot of safe passes..."
        """
        if not self.low_priority_buffer:
            return None
        # Basic stats
        total = len(self.low_priority_buffer)
        passes = sum(1 for _, e, _ in self.low_priority_buffer if e.get("type") in ("pass", "correct_pass"))
        wrong = sum(1 for _, e, _ in self.low_priority_buffer if e.get("type") in ("wrong_pass", "turnover"))
        shots = sum(1 for _, e, _ in self.low_priority_buffer if e.get("type") == "shot")

        text = f"In recent play: {passes} passes, {wrong} turnovers and {shots} shots. Keepers and defenders staying busy."
        self.low_priority_buffer.clear()
        return text

    def _update_best_worst(self, event: Dict[str, Any]):
        """Maintain top-K best events (goals, big xG) and worst events (costly turnovers)"""
        score = event.get('_importance_score', 0)
        # Best events heap (min-heap): keep top K by score
        if len(self.best_events_heap) < self.top_k:
            heapq.heappush(self.best_events_heap, (score, event))
        else:
            if score > self.best_events_heap[0][0]:
                heapq.heapreplace(self.best_events_heap, (score, event))

        # Worst events: focus on turnovers that allowed opponent chance
        if event.get("type") in ("wrong_pass", "turnover") and event.get("leads_to_shot"):
            wscore = -score  # more negative = worse
            if len(self.worst_events_heap) < self.top_k:
                heapq.heappush(self.worst_events_heap, (wscore, event))
            else:
                if wscore < self.worst_events_heap[0][0]:
                    heapq.heapreplace(self.worst_events_heap, (wscore, event))

    def _frame_to_minute(self, frame, fps=30):
        if frame is None: return None
        total_seconds = frame / fps
        minute = int(total_seconds // 60)
        return minute

    # -------------------------
    # SUMMARY GENERATION (HALF / FULL)
    # -------------------------
    def produce_summary(self, when="halftime") -> str:
        """
        Produce a textual summary for halftime/fulltime based on best/worst events & simple tactical insights.
        Returns the full text ready for TTS.
        """
        lines = []
        # Header
        lines.append(f"Half-time summary." if when == "halftime" else "Full-time summary.")
        # Best events
        lines.append(SUMMARY_TEMPLATES["best_events_intro"][0])
        bests = sorted(self.best_events_heap, key=lambda x: -x[0])
        if bests:
            for score, ev in bests:
                lines.append(self._format_event_for_summary(ev))
        else:
            lines.append("No standout positive moments to report.")

        # Worst events
        lines.append(SUMMARY_TEMPLATES["worst_events_intro"][0])
        worsts = sorted(self.worst_events_heap, key=lambda x: x[0])  # already negative ordering
        if worsts:
            for _, ev in worsts:
                lines.append(self._format_event_for_summary(ev))
        else:
            lines.append("No costly errors to report.")

        # Tactical takeaways (very simple automated heuristics)
        lines.append(SUMMARY_TEMPLATES["tactical_summary_intro"][0])
        tactical_lines = self._generate_tactical_takeaways()
        lines.extend(tactical_lines)

        # Closing
        lines.append(SUMMARY_TEMPLATES["closing"][0])
        summary_text = " ".join(lines)
        return summary_text

    def _format_event_for_summary(self, ev: Dict[str, Any]) -> str:
        typ = ev.get("type", ev.get("event", "event"))
        minute = self._frame_to_minute(ev.get("frame")) or "unknown time"
        if typ in ("goal",):
            return f"{ev.get('player_name', 'A player')} scored at minute {minute} — a critical moment."
        if typ == "shot":
            return f"Shot by {ev.get('player_name', 'A player')} at minute {minute} with xG {ev.get('xg', 0):.2f}."
        if typ in ("wrong_pass", "turnover"):
            return f"Turnover by Player {ev.get('player_from')} at minute {minute} led to a danger chance."
        return f"{typ.replace('_', ' ').title()} at minute {minute}."

    def _generate_tactical_takeaways(self) -> List[str]:
        """
        Very simple auto-tactical analysis derived from events.
        You can extend this with your PatternRecogniser / PitchControlModel outputs.
        """
        # Example heuristics:
        total_events = len(self.all_events)
        goals = sum(1 for e in self.all_events if e.get("type") == "goal")
        turnovers = sum(1 for e in self.all_events if e.get("type") in ("wrong_pass", "turnover"))
        pressing_traps = sum(1 for e in self.all_events if e.get("type") == "pressing_trap_success")

        lines = []
        if pressing_traps > 0:
            lines.append(f"The team successfully executed pressing traps {pressing_traps} times; high intensity pressing worked.")
        if turnovers > max(3, total_events*0.02):
            lines.append(f"Turnovers were frequent ({turnovers}) — improve ball security under pressure.")
        if goals == 0 and total_events > 20:
            lines.append("Despite possession, no goals were scored — consider more direct penetration in the final third.")
        if not lines:
            lines.append("No major tactical concerns detected from the event stream.")
        return lines