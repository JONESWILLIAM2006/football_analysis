# 🎙️ Live Commentary System Setup

## Quick Installation

### Option 1: Offline TTS (Recommended)
```bash
pip install pyttsx3
```

### Option 2: Online TTS (Fallback)
```bash
pip install gtts pygame
```

### Option 3: Both (Best Experience)
```bash
pip install pyttsx3 gtts pygame
```

## Features Added

### 🔴 Live Commentary
- **Real-time event commentary** during video analysis
- **Intelligent prioritization** - goals and shots get immediate commentary
- **Rate limiting** - prevents overlapping speech
- **Event aggregation** - summarizes low-priority events periodically

### 📊 Match Summary
- **Automatic summaries** at halftime/fulltime
- **Best/worst moments** analysis
- **Tactical insights** generation
- **Configurable commentary gap** (2-10 seconds)

### 🎯 Event Types Supported
- Goals (highest priority)
- Shots on/off target
- Passes (correct/wrong)
- Fouls and cards
- Tactical patterns
- Ball possession changes

## Usage

1. **Enable in Interface**: Check "Enable Live Commentary" in the ball detection settings
2. **Adjust Settings**: Set commentary gap based on preference
3. **Start Analysis**: Commentary will automatically begin during video processing
4. **Test System**: Use the "Test Commentary System" expander to verify setup

## Technical Details

### Audio Backends
- **pyttsx3**: Offline, low-latency, best for real-time
- **gTTS**: Online, higher quality, requires internet
- **pygame**: For audio playback fallback

### Commentary Logic
- Events scored 0-10 for importance
- High priority (6+): Immediate commentary
- Medium priority (2-6): Queued commentary
- Low priority (<2): Aggregated summaries

### Performance
- Minimal CPU overhead
- Background threading for audio
- Non-blocking event processing
- Automatic cleanup on completion

## Troubleshooting

### No Audio Output
1. Check system audio settings
2. Verify TTS backend installation
3. Test with commentary test interface

### Commentary Too Frequent
- Increase "Commentary Gap" setting
- Events are automatically rate-limited

### Missing Dependencies
```bash
# Install all dependencies
pip install pyttsx3 gtts pygame streamlit ultralytics opencv-python
```

## Integration Points

The commentary system integrates with:
- Ball detection events
- Player tracking
- Tactical analysis
- Match statistics
- Event scoring system

Commentary is automatically triggered for all major match events and provides an immersive analysis experience.