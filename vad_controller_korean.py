"""
WebRTC VAD (Voice Activity Detection) 기반 자동 녹음 제어기
"""

import webrtcvad
import collections
from itertools import islice
import logging


class VADController:
    """WebRTC VAD 기반 자동 녹음 제어기

    음성 활동 감지를 통해 녹음을 자동으로 일시정지/재개/종료합니다.
    """

    # 상태 정의
    STATE_IDLE = 0       # 대기 상태
    STATE_RECORDING = 1  # 녹음 중
    STATE_PAUSED = 2     # 일시정지

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
            sample_rate: 샘플링 레이트 (8000, 16000, 32000, 48000 중 하나)
            frame_ms: 프레임 길이 (ms) - 10, 20, 30 중 하나
            aggr: VAD 민감도 (0-3, 0=관대, 3=엄격)
            start_window_ms: 녹음 시작/재개 판단 윈도우 (ms)
            pause_window_ms: 일시정지 판단 윈도우 (ms)
            stop_window_ms: 자동 종료 판단 윈도우 (ms)
            voice_ratio: 음성으로 판단할 최소 비율 (0.0-1.0)
            silence_ratio: 무음으로 판단할 최소 비율 (0.0-1.0)
            check_interval: 비율 계산 주기 (프레임 수)
            prebuffer_ms: Pre-buffer 길이 (ms)
            grace_period_sec: 유예 기간 (초) - 이 시간 동안은 자동 일시정지 안 됨
        """
        # WebRTC VAD 지원 샘플레이트 검증
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"Unsupported sample rate: {sample_rate}. Must be 8000, 16000, 32000, or 48000")

        # 프레임 길이 검증
        if frame_ms not in (10, 20, 30):
            raise ValueError(f"Unsupported frame length: {frame_ms}. Must be 10, 20, or 30")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.samples_per_frame = int(sample_rate * frame_ms / 1000)

        # VAD 엔진 초기화
        self.vad = webrtcvad.Vad(aggr)
        logging.info(f"VAD initialized: rate={sample_rate}Hz, frame={frame_ms}ms, aggr={aggr}")

        # 상태
        self.state = self.STATE_IDLE
        self.recording_start_time = None
        self.grace_period_sec = grace_period_sec

        # 윈도우 크기 (프레임 단위로 변환)
        self.start_window_frames = max(1, int(start_window_ms / frame_ms))
        self.pause_window_frames = max(1, int(pause_window_ms / frame_ms))
        self.stop_window_frames = max(1, int(stop_window_ms / frame_ms))
        self.prebuffer_frames = int(prebuffer_ms / frame_ms)

        # 임계값
        self.voice_ratio_threshold = voice_ratio
        self.silence_ratio_threshold = silence_ratio
        self.check_interval = check_interval

        # 버퍼 - 최대 윈도우 크기만큼 유지
        max_window = max(self.start_window_frames,
                        self.pause_window_frames,
                        self.stop_window_frames)
        self.sliding_window = collections.deque(maxlen=max_window)
        self.prebuffer = collections.deque(maxlen=self.prebuffer_frames)

        # 카운터
        self.frame_counter = 0
        self.pause_count = 0

        logging.info(f"VAD windows: start={start_window_ms}ms, pause={pause_window_ms}ms, stop={stop_window_ms}ms")
        logging.info(f"VAD thresholds: voice={voice_ratio}, silence={silence_ratio}")

    def start_session(self, current_time):
        """녹음 세션 시작 (녹음 시작 시간 기록)

        Args:
            current_time: 현재 시간 (time.time() 값)
        """
        self.recording_start_time = current_time
        self.state = self.STATE_IDLE
        logging.info(f"VAD session started with {self.grace_period_sec}s grace period")

    def is_in_grace_period(self, current_time):
        """유예 기간 내인지 확인

        Args:
            current_time: 현재 시간 (time.time() 값)

        Returns:
            bool: 유예 기간 내이면 True
        """
        if self.recording_start_time is None:
            return False

        elapsed = current_time - self.recording_start_time
        return elapsed < self.grace_period_sec

    def process_frame(self, frame_bytes, current_time):
        """오디오 프레임을 처리하고 상태 전환을 판단합니다.

        Args:
            frame_bytes: int16 PCM 데이터 (self.samples_per_frame * 2 바이트)
            current_time: 현재 시간 (time.time() 값)

        Returns:
            dict: {
                'state': 현재 상태 (STATE_IDLE/RECORDING/PAUSED),
                'action': 'start' | 'pause' | 'resume' | 'stop' | None,
                'prebuffer': prebuffer 데이터 (action='start'/'resume'시),
                'should_record': 현재 프레임을 녹음할지 여부,
                'pause_count': 총 일시정지 횟수,
                'in_grace_period': 유예 기간 내 여부
            }
        """
        # 프레임 크기 검증
        expected_size = self.samples_per_frame * 2  # int16 = 2 bytes
        if len(frame_bytes) != expected_size:
            logging.warning(f"Frame size mismatch: expected {expected_size}, got {len(frame_bytes)}")
            # 크기가 맞지 않으면 VAD 판정을 건너뛰고 녹음만 수행
            return {
                'state': self.state,
                'action': None,
                'prebuffer': None,
                'should_record': True,
                'pause_count': self.pause_count,
                'in_grace_period': self.is_in_grace_period(current_time)
            }

        # VAD 판정
        try:
            is_voiced = self.vad.is_speech(frame_bytes, self.sample_rate)
        except Exception as e:
            logging.error(f"VAD processing error: {e}")
            # VAD 오류 시 음성으로 간주
            is_voiced = True

        self.sliding_window.append(is_voiced)
        self.frame_counter += 1

        # 주기적 체크 여부
        should_check = (self.frame_counter % self.check_interval == 0)

        # 결과 초기화
        action = None
        prebuffer_data = None
        should_record = False
        in_grace_period = self.is_in_grace_period(current_time)

        # 상태별 처리
        if self.state == self.STATE_IDLE:
            # IDLE 상태: prebuffer에만 저장
            self.prebuffer.append(frame_bytes)

            # 녹음 시작 조건 확인
            if should_check and len(self.sliding_window) >= self.start_window_frames:
                voice_ratio = self._calc_voice_ratio(self.start_window_frames)

                if voice_ratio >= self.voice_ratio_threshold:
                    # 녹음 시작
                    self.state = self.STATE_RECORDING
                    action = 'start'
                    prebuffer_data = list(self.prebuffer)
                    self.prebuffer.clear()
                    should_record = True
                    logging.info(f"VAD: Recording started (voice ratio: {voice_ratio:.2f})")

        elif self.state == self.STATE_RECORDING:
            # RECORDING 상태: 항상 녹음
            should_record = True
            self.prebuffer.append(frame_bytes)

            # 유예 기간이 지난 후에만 일시정지 체크
            if not in_grace_period and should_check and len(self.sliding_window) >= self.pause_window_frames:
                voice_ratio = self._calc_voice_ratio(self.pause_window_frames)
                silence_ratio = 1.0 - voice_ratio

                if silence_ratio >= self.silence_ratio_threshold:
                    # 일시정지
                    self.state = self.STATE_PAUSED
                    action = 'pause'
                    self.pause_count += 1
                    should_record = False
                    logging.info(f"VAD: Paused #{self.pause_count} (silence ratio: {silence_ratio:.2f})")

        elif self.state == self.STATE_PAUSED:
            # PAUSED 상태: prebuffer에만 저장
            self.prebuffer.append(frame_bytes)

            # 재개 확인
            if should_check and len(self.sliding_window) >= self.start_window_frames:
                voice_ratio_short = self._calc_voice_ratio(self.start_window_frames)

                if voice_ratio_short >= self.voice_ratio_threshold:
                    # 재개
                    self.state = self.STATE_RECORDING
                    action = 'resume'
                    prebuffer_data = list(self.prebuffer)
                    self.prebuffer.clear()
                    should_record = True
                    logging.info(f"VAD: Resumed (voice ratio: {voice_ratio_short:.2f})")

            # 종료 확인
            if should_check and len(self.sliding_window) >= self.stop_window_frames:
                voice_ratio_long = self._calc_voice_ratio(self.stop_window_frames)
                silence_ratio_long = 1.0 - voice_ratio_long

                if silence_ratio_long >= self.silence_ratio_threshold:
                    # 종료
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
        """슬라이딩 윈도우에서 음성 비율 계산

        Args:
            num_frames: 계산할 프레임 수

        Returns:
            float: 음성 비율 (0.0-1.0)
        """
        window_len = len(self.sliding_window)
        if window_len < num_frames:
            return 0.0

        # 최근 num_frames만큼 슬라이스
        start_idx = window_len - num_frames
        voice_count = sum(islice(self.sliding_window, start_idx, None))
        return voice_count / num_frames

    def _reset(self):
        """내부 상태 리셋 (카운터 및 버퍼 초기화)"""
        self.sliding_window.clear()
        self.prebuffer.clear()
        self.frame_counter = 0
        # pause_count는 세션 통계이므로 유지

    def force_stop(self):
        """강제 종료 (수동 종료 시 호출)

        녹음 세션을 완전히 종료하고 모든 상태를 초기화합니다.
        """
        self.state = self.STATE_IDLE
        self.recording_start_time = None
        self._reset()
        self.pause_count = 0
        logging.info("VAD: Force stopped and reset")

    def get_state_name(self):
        """현재 상태의 이름 반환

        Returns:
            str: 상태 이름 ("대기", "녹음중", "일시정지")
        """
        state_names = {
            self.STATE_IDLE: "대기",
            self.STATE_RECORDING: "녹음중",
            self.STATE_PAUSED: "일시정지"
        }
        return state_names.get(self.state, "알 수 없음")

    def get_stats(self):
        """통계 정보 반환

        Returns:
            dict: 통계 정보 (상태, 일시정지 횟수, 프레임 수 등)
        """
        return {
            'state': self.get_state_name(),
            'pause_count': self.pause_count,
            'frame_counter': self.frame_counter,
            'window_size': len(self.sliding_window),
            'prebuffer_size': len(self.prebuffer)
        }
