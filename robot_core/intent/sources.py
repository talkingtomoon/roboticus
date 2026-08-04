"""의도 입력 소스 — "사람의 말"이 텍스트 발화로 나오는 곳.

발화(str)를 산출하는 것이 소스의 전부다. 해석은 interpreter가, 주입은
Supervisor.request_intent()(가드 경유)가 한다 — 소스는 텍스트만 만든다.

- TypedSource   : 큐에 문자열을 넣으면 발화로 나온다 (웹 UI/테스트의 입력)
- WhisperSource : faster-whisper 로컬 STT 스켈레톤. 모델 로드는 구현돼 있고
                  마이크 캡처는 현장 몫 (하드웨어 접근 코드 금지 원칙)
"""

from __future__ import annotations

import queue
from dataclasses import dataclass


@dataclass
class SourceConfig:
    language: str = "ko"
    silence_end_s: float = 0.8     # 발화 종료 판정: 무음 N초
    min_length: int = 2            # 이보다 짧은 발화는 버린다 (숨소리/오탐)


class IntentSourceUnavailable(RuntimeError):
    """소스를 쓸 수 없는 사유 (의존성 미설치 등). 메시지에 해결법 포함."""


class IntentSource:
    """발화 소스 계약."""

    def get(self, timeout: float | None = None) -> str | None:
        """다음 발화. timeout 안에 없으면 None."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class TypedSource(IntentSource):
    """타이핑 입력 (웹 UI의 텍스트창, 테스트). put() → get()."""

    def __init__(self, config: SourceConfig | None = None) -> None:
        self.cfg = config or SourceConfig()
        self._q: queue.Queue[str] = queue.Queue(maxsize=64)
        self.dropped_short = 0

    def put(self, text: str) -> bool:
        text = str(text).strip()
        if len(text) < self.cfg.min_length:
            self.dropped_short += 1
            return False
        try:
            self._q.put_nowait(text)
            return True
        except queue.Full:
            return False

    def get(self, timeout: float | None = None) -> str | None:
        try:
            if timeout is None:
                return self._q.get_nowait()
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class WhisperSource(IntentSource):
    """faster-whisper 로컬 STT — 스켈레톤.

    Jetson에서 `pip install faster-whisper` 후 활성화되는 구조:
    - 모델 로드는 여기 구현돼 있다 (__init__에서 실제로 로드한다)
    - 마이크 캡처 루프는 현장에서 채운다 (아래 스켈레톤 주석) —
      하드웨어 접근 코드는 이 저장소에 넣지 않는다. 장치는 문자열로만.

    ------------------------------------------------------------------
    현장(Jetson) 실측 항목 — 채우면서 기록할 것:
    ------------------------------------------------------------------
    1) 모델 크기별 지연 (발화 3초 기준, 1회 전사 wall time):
         tiny:   ___ ms     (품질 낮음, 짧은 명령어면 충분할 수도)
         base:   ___ ms
         small:  ___ ms     (기본값 — 한국어 품질/지연 균형 예상)
         medium: ___ ms     (지연이 3초 넘으면 안전 명령엔 부적합)
       → 안전 명령("멈춰")은 어차피 해석기 키워드 경로가 받으므로
         STT 지연 기준은 '일반 명령 체감'으로 잡으면 된다.
    2) 마이크 장치: `python -m sounddevice` 로 목록 확인 후
       device_name에 문자열로 지정 (인덱스는 재부팅에 흔들린다)
    3) VAD: silence_end_s(무음 종료)가 행사장 소음에서 오작동하는지 —
       faster-whisper vad_filter=True + min_silence_duration_ms로 조정

    캡처 루프 스켈레톤 (현장에서 채움):
    # import sounddevice as sd          # pip install sounddevice
    # def _capture_loop(self):
    #     with sd.InputStream(device=self.device_name, samplerate=16000,
    #                         channels=1, dtype="float32") as stream:
    #         buf = []
    #         while not self._stop.is_set():
    #             chunk, _ = stream.read(1600)          # 0.1s
    #             buf.append(chunk)
    #             if self._silence_detected(buf):        # cfg.silence_end_s
    #                 audio = np.concatenate(buf); buf = []
    #                 segs, _ = self._model.transcribe(
    #                     audio[:, 0], language=self.cfg.language,
    #                     vad_filter=True)
    #                 text = " ".join(s.text for s in segs).strip()
    #                 if len(text) >= self.cfg.min_length:
    #                     self._q.put(text)
    """

    def __init__(self, model_size: str = "small", device_name: str = "default",
                 config: SourceConfig | None = None) -> None:
        self.cfg = config or SourceConfig()
        self.model_size = model_size
        self.device_name = str(device_name)   # 장치 지정은 문자열로만
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise IntentSourceUnavailable(
                "faster-whisper가 없다 — Jetson에서 `pip install faster-whisper` "
                "후 사용 가능. 그때까지 TypedSource(텍스트 입력)로 동작한다."
            ) from e
        # 모델 로드는 실제로 한다 (다운로드 포함 수십 초 걸릴 수 있음)
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def get(self, timeout: float | None = None) -> str | None:
        raise NotImplementedError(
            "마이크 캡처 루프는 현장에서 채운다 — 클래스 독스트링의 스켈레톤과 "
            "실측 항목 참고. 장치는 device_name 문자열로 지정.")
