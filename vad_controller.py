"""
WebRTC VAD (Voice Activity Detection)–based automatic recording controller.

This module integrates with dnote_main.py's recording logic and provides
automatic pause/resume/stop behaviors driven by speech activity.

Goal: keep STT input focused on speech segments to reduce hallucinations
caused by long silent periods.
"""

import webrtcvad
import collections
from itertools import islice
import logging


class VADController:
    """Automatic recorder controller based on WebRTC VAD.

    Uses voice activity to automatically pause, resume, and stop recording.
    """

    # State definitions
    STATE_IDLE = 0       # Idle (not currently recording; collecting prebuffer only)
    STATE_RECORDING = 1  # Actively recording (should write frames)
    STATE_PAUSED = 2     # Temporarily paused (collecting prebuffer only)

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
            sample_rate: Sampling rate in Hz (one of 8000, 16000, 32000, 48000).
            frame_ms: Frame length in milliseconds (one of 10, 20, 30).
            aggr: VAD aggressiveness (0–3; 0 = lenient, 3 = strict).
            start_window_ms: Window (ms) for deciding start/resume.
            pause_window_ms: Window (ms) for deciding pause.
            stop_window_ms: Window (ms) for deciding automatic stop.
            voice_ratio: Minimum voiced ratio (0.0–1.0) to treat a window as speech.
            silence_ratio: Minimum unvoiced ratio (0.0–1.0) to treat a window as silence.
            check_interval: Compute ratios every N frames (performance/anti-flutter).
            prebuffer_ms: Prebuffer length (ms) to prepend on start/resume to avoid clipping onsets.
            grace_period_sec: Grace period (s) after session start during which auto-pause is disabled.
        """
        # Validate WebRTC VAD supported sample rates
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(
                f"Unsupported sample rate: {sample_rate}. Must be 8000, 16000, 32000, or 48000"
            )

        # Validate frame length
        if frame_ms not in (10, 20, 30):
            raise ValueError(
                f"Unsupported frame length: {frame_ms}. Must be 10, 20, or 30"
            )

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

        # Convert windows from ms to frame counts
        self.start_window_frames = max(1, int(start_window_ms / frame_ms))
        self.pause_window_frames = max(1, int(pause_window_ms / frame_ms))
        self.stop_window_frames = max(1, int(stop_window_ms / frame_ms))
        self.prebuffer_frames = int(prebuffer_ms / frame_ms)

        # Thresholds and cadence
        self.voice_ratio_threshold = voice_ratio
        self.silence_ratio_threshold = silence_ratio
        self.check_interval = check_interval

        # Buffers: keep up to the maximum needed window size
        max_window = max(
            self.start_window_frames,
            self.pause_window_frames,
            self.stop_window_frames
        )
        self.sliding_window = collections.deque(maxlen=max_window)
        self.prebuffer = collections.deque(maxlen=self.prebuffer_frames)

        # Counters
        self.frame_counter = 0
        self.pause_count = 0

        logging.info(
            f"VAD windows (ms): start={start_window_ms}, pause={pause_window_ms}, stop={stop_window_ms}"
        )
        logging.info(
            f"VAD thresholds: voice_ratio={voice_ratio}, silence_ratio={silence_ratio}"
        )

    def start_session(self, current_time):
        """Begin a recording session (records session start time).

        Args:
            current_time: Current wall clock (e.g., time.time()).
        """
        self.recording_start_time = current_time
        self.state = self.STATE_IDLE
        logging.info(f"VAD session started with a {self.grace_period_sec}s grace period")

    def is_in_grace_period(self, current_time):
        """Check whether we are still within the initial grace period.

        Args:
            current_time: Current wall clock (e.g., time.time()).

        Returns:
            bool: True if within grace period.
        """
        if self.recording_start_time is None:
            return False

        elapsed = current_time - self.recording_start_time
        return elapsed < self.grace_period_sec

    def process_frame(self, frame_bytes, current_time):
        """Process one PCM frame and decide state transitions.

        Args:
            frame_bytes: int16 PCM (little-endian) mono; length must be samples_per_frame * 2.
            current_time: Current wall clock (e.g., time.time()).

        Returns:
            dict: {
                'state': current state (STATE_IDLE / STATE_RECORDING / STATE_PAUSED),
                'action': 'start' | 'pause' | 'resume' | 'stop' | None,
                'prebuffer': list of frames to prepend on 'start'/'resume', else None,
                'should_record': whether to write this frame,
                'pause_count': total number of pauses so far,
                'in_grace_period': whether grace period is active
            }
        """
        # Validate frame size
        expected_size = self.samples_per_frame * 2  # int16 = 2 bytes
        if len(frame_bytes) != expected_size:
            logging.warning(f"Frame size mismatch: expected {expected_size}, got {len(frame_bytes)}")
            # If size is off, skip VAD decision but keep recording to avoid data loss
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
            # On VAD errors, treat as voiced to avoid dropping audio
            is_voiced = True

        self.sliding_window.append(is_voiced)
        self.frame_counter += 1

        # Only compute ratios periodically
        should_check = (self.frame_counter % self.check_interval == 0)

        # Initialize outputs
        action = None
        prebuffer_data = None
        should_record = False
        in_grace_period = self.is_in_grace_period(current_time)

        # State handling
        if self.state == self.STATE_IDLE:
            # In IDLE: keep collecting prebuffer only
            self.prebuffer.append(frame_bytes)

            # Check start/resume condition
            if should_check and len(self.sliding_window) >= self.start_window_frames:
                voice_ratio = self._calc_voice_ratio(self.start_window_frames)

                if voice_ratio >= self.voice_ratio_threshold:
                    # Transition to RECORDING
                    self.state = self.STATE_RECORDING
                    action = 'start'
                    prebuffer_data = list(self.prebuffer)
                    self.prebuffer.clear()
                    should_record = True
                    logging.info(f"VAD: Recording started (voice ratio: {voice_ratio:.2f})")

        elif self.state == self.STATE_RECORDING:
            # In RECORDING: always record
            should_record = True
            self.prebuffer.append(frame_bytes)

            # After grace period, check for pause
            if (not in_grace_period) and should_check and len(self.sliding_window) >= self.pause_window_frames:
                voice_ratio = self._calc_voice_ratio(self.pause_window_frames)
                silence_ratio = 1.0 - voice_ratio

                if silence_ratio >= self.silence_ratio_threshold:
                    # Transition to PAUSED
                    self.state = self.STATE_PAUSED
                    action = 'pause'
                    self.pause_count += 1
                    should_record = False
                    logging.info(f"VAD: Paused #{self.pause_count} (silence ratio: {silence_ratio:.2f})")

        elif self.state == self.STATE_PAUSED:
            # In PAUSED: collect into prebuffer
            self.prebuffer.append(frame_bytes)

            # Check for resume
            if should_check and len(self.sliding_window) >= self.start_window_frames:
                voice_ratio_short = self._calc_voice_ratio(self.start_window_frames)

                if voice_ratio_short >= self.voice_ratio_threshold:
                    # Transition back to RECORDING
                    self.state = self.STATE_RECORDING
                    action = 'resume'
                    prebuffer_data = list(self.prebuffer)
                    self.prebuffer.clear()
                    should_record = True
                    logging.info(f"VAD: Resumed (voice ratio: {voice_ratio_short:.2f})")

            # Check for long-stop
            if should_check and len(self.sliding_window) >= self.stop_window_frames:
                voice_ratio_long = self._calc_voice_ratio(self.stop_window_frames)
                silence_ratio_long = 1.0 - voice_ratio_long

                if silence_ratio_long >= self.silence_ratio_threshold:
                    # Auto-stop back to IDLE
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
        """Compute voiced ratio over the last `num_frames` in the sliding window.

        Args:
            num_frames: number of frames to consider.

        Returns:
            float: voiced ratio in [0.0, 1.0].
        """
        window_len = len(self.sliding_window)
        if window_len < num_frames:
            return 0.0

        # Slice the most recent num_frames
        start_idx = window_len - num_frames
        voice_count = sum(islice(self.sliding_window, start_idx, None))
        return voice_count / num_frames

    def _reset(self):
        """Reset internal counters and buffers (pause_count is preserved as session stats)."""
        self.sliding_window.clear()
        self.prebuffer.clear()
        self.frame_counter = 0
        # Note: pause_count is kept.

    def force_stop(self):
        """Force a full stop (manual termination of the session).

        Resets recording session and all internal state.
        """
        self.state = self.STATE_IDLE
        self.recording_start_time = None
        self._reset()
        self.pause_count = 0
        logging.info("VAD: Force stopped and reset")

    def get_state_name(self):
        """Return the current state's display name (Korean strings kept for UI compatibility).

        Returns:
            str: "대기" (idle), "녹음중" (recording), or "일시정지" (paused).
        """
        state_names = {
            self.STATE_IDLE: "대기",
            self.STATE_RECORDING: "녹음중",
            self.STATE_PAUSED: "일시정지"
        }
        return state_names.get(self.state, "알 수 없음")

    def get_stats(self):
        """Return statistics for monitoring/telemetry.

        Returns:
            dict: includes display state name, pause count, frame counter, and buffer sizes.
        """
        return {
            'state': self.get_state_name(),
            'pause_count': self.pause_count,
            'frame_counter': self.frame_counter,
            'window_size': len(self.sliding_window),
            'prebuffer_size': len(self.prebuffer)
        }
