"""
WebRTC VAD (Voice Activity Detection) Based Automatic Recording Controller

This module integrates with dnote_main.py's recording functionality to provide
automatic pause/resume/stop features based on voice activity detection.
"""

import webrtcvad
import collections
from itertools import islice
import logging


class VADController:
    """WebRTC VAD-based Automatic Recording Controller

    Automatically pauses/resumes/stops recording based on voice activity detection.
    """

    # State definitions
    STATE_IDLE = 0       # Idle state
    STATE_RECORDING = 1  # Recording
    STATE_PAUSED = 2     # Paused

    def __init__(self,
                 sample_rate=32000,
                 frame_ms=30,
                 aggr=2,
                 start_window_ms=600,
                 pause_window_ms=12000,
                 stop_window_ms=60000,
                 voice_ratio=0.7,
                 silence_ratio=0.92,
                 check_interval=10,
                 prebuffer_ms=1200,
                 grace_period_sec=20):
        """
        Args:
            sample_rate: Sample rate (must be 8000, 16000, 32000, or 48000)
            frame_ms: Frame length in ms (must be 10, 20, or 30)
            aggr: VAD aggressiveness (0-3, 0=lenient, 3=aggressive)
            start_window_ms: Window for start/resume decision (ms)
            pause_window_ms: Window for pause decision (ms)
            stop_window_ms: Window for auto-stop decision (ms)
            voice_ratio: Minimum ratio to detect as voice (0.0-1.0)
            silence_ratio: Minimum ratio to detect as silence (0.0-1.0)
            check_interval: Decision check interval (number of frames)
            prebuffer_ms: Pre-buffer length (ms)
            grace_period_sec: Grace period (seconds) - no auto-pause during this time
        """
        # Validate WebRTC VAD supported sample rates
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"Unsupported sample rate: {sample_rate}. Must be 8000, 16000, 32000, or 48000")

        # Validate frame length
        if frame_ms not in (10, 20, 30):
            raise ValueError(f"Unsupported frame length: {frame_ms}. Must be 10, 20, or 30")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.samples_per_frame = int(sample_rate * frame_ms / 1000)

        # Initialize VAD engine
        self.vad = webrtcvad.Vad(aggr)
        logging.info(f"VAD initialized: rate={sample_rate}Hz, frame={frame_ms}ms, aggr={aggr}")

        # State
        self.state = self.STATE_IDLE
        self.recording_start_time = None
        self.grace_period_sec = grace_period_sec

        # Window sizes (converted to frame units)
        self.start_window_frames = max(1, int(start_window_ms / frame_ms))
        self.pause_window_frames = max(1, int(pause_window_ms / frame_ms))
        self.stop_window_frames = max(1, int(stop_window_ms / frame_ms))
        self.prebuffer_frames = int(prebuffer_ms / frame_ms)

        # Thresholds
        self.voice_ratio_threshold = voice_ratio
        self.silence_ratio_threshold = silence_ratio
        self.check_interval = check_interval

        # Buffers - maintain up to max window size
        max_window = max(self.start_window_frames,
                        self.pause_window_frames,
                        self.stop_window_frames)
        self.sliding_window = collections.deque(maxlen=max_window)
        self.prebuffer = collections.deque(maxlen=self.prebuffer_frames)

        # Counters
        self.frame_counter = 0
        self.pause_count = 0

        logging.info(f"VAD windows: start={start_window_ms}ms, pause={pause_window_ms}ms, stop={stop_window_ms}ms")
        logging.info(f"VAD thresholds: voice={voice_ratio}, silence={silence_ratio}")

    def start_session(self, current_time):
        """Start recording session (record start time)

        Args:
            current_time: Current time (time.time() value)
        """
        self.recording_start_time = current_time
        self.state = self.STATE_IDLE
        logging.info(f"VAD session started with {self.grace_period_sec}s grace period")

    def is_in_grace_period(self, current_time):
        """Check if within grace period

        Args:
            current_time: Current time (time.time() value)

        Returns:
            bool: True if within grace period
        """
        if self.recording_start_time is None:
            return False

        elapsed = current_time - self.recording_start_time
        return elapsed < self.grace_period_sec

    def process_frame(self, frame_bytes, current_time):
        """Process audio frame and determine state transitions.

        Args:
            frame_bytes: int16 PCM data (self.samples_per_frame * 2 bytes)
            current_time: Current time (time.time() value)

        Returns:
            dict: {
                'state': Current state (STATE_IDLE/RECORDING/PAUSED),
                'action': 'start' | 'pause' | 'resume' | 'stop' | None,
                'prebuffer': Prebuffer data (when action='start'/'resume'),
                'should_record': Whether to record current frame,
                'pause_count': Total pause count,
                'in_grace_period': Whether in grace period
            }
        """
        # Validate frame size
        expected_size = self.samples_per_frame * 2  # int16 = 2 bytes
        if len(frame_bytes) != expected_size:
            logging.warning(f"Frame size mismatch: expected {expected_size}, got {len(frame_bytes)}")
            # If size doesn't match, skip VAD and just record
            return {
                'state': self.state,
                'action': None,
                'prebuffer': None,
                'should_record': True,
                'pause_count': self.pause_count,
                'in_grace_period': self.is_in_grace_period(current_time)
            }

        # VAD decision
        try:
            is_voiced = self.vad.is_speech(frame_bytes, self.sample_rate)
        except Exception as e:
            logging.error(f"VAD processing error: {e}")
            # On VAD error, assume voice
            is_voiced = True

        self.sliding_window.append(is_voiced)
        self.frame_counter += 1

        # Check if periodic check needed
        should_check = (self.frame_counter % self.check_interval == 0)

        # Initialize results
        action = None
        prebuffer_data = None
        should_record = False
        in_grace_period = self.is_in_grace_period(current_time)

        # Process by state
        if self.state == self.STATE_IDLE:
            # IDLE state: only store in prebuffer
            self.prebuffer.append(frame_bytes)

            # Check recording start condition
            if should_check and len(self.sliding_window) >= self.start_window_frames:
                voice_ratio = self._calc_voice_ratio(self.start_window_frames)

                if voice_ratio >= self.voice_ratio_threshold:
                    # Start recording
                    self.state = self.STATE_RECORDING
                    action = 'start'
                    prebuffer_data = list(self.prebuffer)
                    self.prebuffer.clear()
                    should_record = True
                    logging.info(f"VAD: Recording started (voice ratio: {voice_ratio:.2f})")

        elif self.state == self.STATE_RECORDING:
            # RECORDING state: always record
            should_record = True
            self.prebuffer.append(frame_bytes)

            # Check pause only after grace period
            if not in_grace_period and should_check and len(self.sliding_window) >= self.pause_window_frames:
                voice_ratio = self._calc_voice_ratio(self.pause_window_frames)
                silence_ratio = 1.0 - voice_ratio

                if silence_ratio >= self.silence_ratio_threshold:
                    # Pause
                    self.state = self.STATE_PAUSED
                    action = 'pause'
                    self.pause_count += 1
                    should_record = False
                    logging.info(f"VAD: Paused #{self.pause_count} (silence ratio: {silence_ratio:.2f})")

        elif self.state == self.STATE_PAUSED:
            # PAUSED state: only store in prebuffer
            self.prebuffer.append(frame_bytes)

            # Check resume
            if should_check and len(self.sliding_window) >= self.start_window_frames:
                voice_ratio_short = self._calc_voice_ratio(self.start_window_frames)

                if voice_ratio_short >= self.voice_ratio_threshold:
                    # Resume
                    self.state = self.STATE_RECORDING
                    action = 'resume'
                    prebuffer_data = list(self.prebuffer)
                    self.prebuffer.clear()
                    should_record = True
                    logging.info(f"VAD: Resumed (voice ratio: {voice_ratio_short:.2f})")

            # Check stop
            if should_check and len(self.sliding_window) >= self.stop_window_frames:
                voice_ratio_long = self._calc_voice_ratio(self.stop_window_frames)
                silence_ratio_long = 1.0 - voice_ratio_long

                if silence_ratio_long >= self.silence_ratio_threshold:
                    # Stop
                    self.state = self.STATE_IDLE
                    action = 'stop'
                    self._reset()
                    logging.info(f"VAD: Auto-stopped (long silence ratio: {silence_ratio_long:.2f})")

        return {
            'state': self.state,
            'action': action,
            'prebuffer': prebuffer_data,
            'should_record': should_record,
            'pause_count': self.pause_count,
            'in_grace_period': in_grace_period
        }

    def _calc_voice_ratio(self, num_frames):
        """Calculate voice ratio in sliding window

        Args:
            num_frames: Number of frames to calculate

        Returns:
            float: Voice ratio (0.0-1.0)
        """
        window_len = len(self.sliding_window)
        if window_len < num_frames:
            return 0.0

        # Slice last num_frames
        start_idx = window_len - num_frames
        voice_count = sum(islice(self.sliding_window, start_idx, None))
        return voice_count / num_frames

    def _reset(self):
        """Reset internal state (counters and buffers)"""
        self.sliding_window.clear()
        self.prebuffer.clear()
        self.frame_counter = 0
        # Keep pause_count as it's session statistics

    def force_stop(self):
        """Force stop (called on manual stop)

        Completely ends the recording session and resets all states.
        """
        self.state = self.STATE_IDLE
        self.recording_start_time = None
        self._reset()
        self.pause_count = 0
        logging.info("VAD: Force stopped and reset")

    def get_state_name(self):
        """Return current state name

        Returns:
            str: State name ("Idle", "Recording", "Paused")
        """
        state_names = {
            self.STATE_IDLE: "Idle",
            self.STATE_RECORDING: "Recording",
            self.STATE_PAUSED: "Paused"
        }
        return state_names.get(self.state, "Unknown")

    def get_stats(self):
        """Return statistics

        Returns:
            dict: Statistics (state, pause count, frame count, etc.)
        """
        return {
            'state': self.get_state_name(),
            'pause_count': self.pause_count,
            'frame_counter': self.frame_counter,
            'window_size': len(self.sliding_window),
            'prebuffer_size': len(self.prebuffer)
        }
