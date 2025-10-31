# WebRTC VAD Controller

A sophisticated Voice Activity Detection (VAD) controller designed to minimize hallucinations in transformer-based Speech-to-Text (STT) models like Whisper by intelligently managing recording based on speech presence.

## Overview

This module provides automatic recording control through voice activity detection, significantly reducing the silent periods that can cause hallucinations in STT models. By recording only during speech periods, it ensures cleaner audio input for transcription systems.

## Key Features

- **Automatic Recording Management**: Automatically starts, pauses, resumes, and stops recording based on voice activity
- **Pre-buffering**: Captures audio before voice detection triggers to avoid cutting off the beginning of speech
- **Grace Period**: Prevents premature pausing during natural speech pauses
- **Configurable Sensitivity**: Adjustable VAD aggressiveness and detection thresholds
- **State Machine Design**: Clean state transitions between IDLE, RECORDING, and PAUSED states
- **Sliding Window Analysis**: Uses configurable time windows for intelligent decision-making

## Why This Matters for STT

Transformer-based STT models like Whisper can generate hallucinations when processing long silent periods. These hallucinations appear as phantom transcriptions of non-existent speech. By using VAD to record only during actual speech:

- ✅ Eliminates silent period hallucinations
- ✅ Reduces file size and processing time
- ✅ Improves transcription accuracy
- ✅ Provides cleaner training data for fine-tuning

## Requirements

```bash
pip install webrtcvad
```

## Quick Start

```python
from vad_controller import VADController
import time

# Initialize controller
vad = VADController(
    sample_rate=32000,
    frame_ms=30,
    aggr=2,  # 0=lenient, 3=aggressive
    start_window_ms=600,
    pause_window_ms=12000,
    stop_window_ms=60000,
    grace_period_sec=20
)

# Start a recording session
vad.start_session(time.time())

# Process audio frames
while recording:
    frame = get_audio_frame()  # Get 30ms of int16 PCM audio
    result = vad.process_frame(frame, time.time())
    
    if result['action'] == 'start':
        print("🎤 Recording started")
        # Write prebuffer to file
        for buffered_frame in result['prebuffer']:
            write_to_file(buffered_frame)
    
    if result['should_record']:
        write_to_file(frame)
    
    if result['action'] == 'pause':
        print("⏸️  Paused (silence detected)")
    
    if result['action'] == 'resume':
        print("▶️  Resumed (voice detected)")
        # Write prebuffer to file
        for buffered_frame in result['prebuffer']:
            write_to_file(buffered_frame)
    
    if result['action'] == 'stop':
        print("⏹️  Auto-stopped (long silence)")
        break
```

## Configuration Parameters

### Sample Rate and Frame Length
- `sample_rate`: Must be 8000, 16000, 32000, or 48000 Hz (WebRTC VAD requirement)
- `frame_ms`: Must be 10, 20, or 30 ms (WebRTC VAD requirement)

### VAD Sensitivity
- `aggr`: VAD aggressiveness (0-3)
  - 0: Lenient (detects more as speech)
  - 1: Moderate
  - 2: Moderate-aggressive (recommended)
  - 3: Aggressive (strict speech detection)

### Time Windows
- `start_window_ms` (default: 600): Time window to confirm speech start/resume
- `pause_window_ms` (default: 12000): Time window to confirm pause condition
- `stop_window_ms` (default: 60000): Time window to confirm auto-stop condition

### Thresholds
- `voice_ratio` (default: 0.7): Minimum ratio of voiced frames to start/resume (0.0-1.0)
- `silence_ratio` (default: 0.92): Minimum ratio of silence to pause/stop (0.0-1.0)

### Other Parameters
- `check_interval` (default: 10): Check decision every N frames
- `prebuffer_ms` (default: 1200): Pre-buffer duration to capture speech onset
- `grace_period_sec` (default: 20): Grace period after recording starts (no auto-pause)

## State Machine

```
┌─────────────┐
│    IDLE     │ ◄──────────────────────────┐
│  (Waiting)  │                            │
└──────┬──────┘                            │
       │ Voice detected                    │
       │ (voice_ratio ≥ threshold)         │
       ▼                                   │
┌─────────────┐                            │
│  RECORDING  │                            │
│  (Active)   │                            │
└──────┬──────┘                            │
       │ Silence detected                  │
       │ (after grace period)              │
       ▼                                   │
┌─────────────┐                            │
│   PAUSED    │                            │
│ (Suspended) │                            │
└──────┬──┬───┘                            │
       │  │                                │
       │  │ Long silence                   │
       │  └────────────────────────────────┘
       │
       │ Voice detected
       └──────────► (Back to RECORDING)
```

## API Reference

### `__init__(...)`
Initialize the VAD controller with configuration parameters.

### `start_session(current_time)`
Start a new recording session. Call this when beginning a recording to initialize the grace period timer.

**Parameters:**
- `current_time`: Current timestamp from `time.time()`

### `process_frame(frame_bytes, current_time)`
Process a single audio frame and determine state transitions.

**Parameters:**
- `frame_bytes`: int16 PCM audio data (must be exactly `samples_per_frame * 2` bytes)
- `current_time`: Current timestamp from `time.time()`

**Returns:** Dictionary containing:
- `state`: Current state (STATE_IDLE/RECORDING/PAUSED)
- `action`: Action triggered ('start', 'pause', 'resume', 'stop', or None)
- `prebuffer`: List of buffered frames (when action is 'start' or 'resume')
- `should_record`: Boolean indicating if current frame should be recorded
- `pause_count`: Total number of pauses in this session
- `in_grace_period`: Boolean indicating if still in grace period

### `force_stop()`
Manually stop and reset the controller. Call this when the user manually stops recording.

### `get_state_name()`
Get human-readable name of current state.

**Returns:** String ("대기", "녹음중", "일시정지")

### `get_stats()`
Get current statistics.

**Returns:** Dictionary with state, pause_count, frame_counter, window_size, prebuffer_size

## Integration Example with Whisper

```python
import whisper
from vad_controller import VADController
import wave
import time

# Initialize
model = whisper.load_model("base")
vad = VADController(sample_rate=16000, frame_ms=30)

# Record with VAD
with wave.open("output.wav", "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(16000)
    
    vad.start_session(time.time())
    
    while recording:
        frame = capture_audio()  # Your audio capture function
        result = vad.process_frame(frame, time.time())
        
        # Handle prebuffer on start/resume
        if result['prebuffer']:
            for buffered_frame in result['prebuffer']:
                wav_file.writeframes(buffered_frame)
        
        # Write current frame if should record
        if result['should_record']:
            wav_file.writeframes(frame)
        
        # Auto-stop on long silence
        if result['action'] == 'stop':
            break

# Transcribe (now with minimal silence)
result = model.transcribe("output.wav")
print(result["text"])
```

## Performance Considerations

- **CPU Usage**: WebRTC VAD is highly efficient and adds minimal overhead
- **Memory**: Sliding window and prebuffer use bounded memory (maxlen on deques)
- **Latency**: Decision latency is `check_interval * frame_ms` (default: 300ms)

## Troubleshooting

### "Frame size mismatch" Warning
Ensure your audio frames are exactly `sample_rate * frame_ms / 1000 * 2` bytes. For 32kHz and 30ms: 1920 bytes.

### Too Sensitive (Frequent Pausing)
- Increase `aggr` (0 → 1 or 2)
- Increase `start_window_ms`
- Increase `pause_window_ms`
- Increase `voice_ratio`
- Increase `silence_ratio`

### Not Sensitive Enough (Doesn't Pause)
- Decrease `aggr` (2 → 1 or 0)
- Decrease `start_window_ms`
- Decrease `pause_window_ms`
- Decrease `voice_ratio`
- Decrease `silence_ratio`

### Missing Speech Onset
- Increase `prebuffer_ms` (e.g., 1800ms)
- Decrease `start_window_ms` for faster response  (e.g., 300ms)
