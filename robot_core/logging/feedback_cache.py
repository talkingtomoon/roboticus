"""FeedbackCache — 1kHz 수신과 2Hz 판단 사이의 유일한 접점.

핵심 규칙(매뉴얼): 1kHz 콜백에서는 최신 상태 저장만. 판단·전송은 느린 루프.
이 클래스가 그 규칙을 구조로 만든다:

- push(frame): 1kHz 콜백이 부른다. deque append + 참조 교체뿐 — 판단 금지
- latest() / to_arrays() / dump_text(): 느린 루프가 부른다

이벤트 마킹(mark_event/events)은 통합 타임라인(integration/timeline.py)이
그대로 소비한다.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from robot_core.hal.phorce import N_AXES, PhorceFeedback


@dataclass
class _Event:
    t: float
    text: str
    seq: int = 0     # 단조 증가 커서 — deque가 오래된 것을 버려도 안 흔들린다


class FeedbackCache:
    def __init__(self, n_axes: int = N_AXES, window_sec: float = 8.0,
                 max_samples: int = 60_000,
                 event_log_path=None) -> None:
        """event_log_path: 주면 mark_event가 JSONL로도 스트리밍한다 (세션 로그).

        메모리 타임라인은 deque(maxlen=300)라 데모 한 번이면 찬다 — 현장
        디버깅("아까 그거 왜 그랬지")은 파일이 담당한다. 한 줄씩 즉시 flush.
        """
        self.n_axes = int(n_axes)
        self.window_sec = float(window_sec)
        self._event_log = None
        if event_log_path is not None:
            from pathlib import Path
            p = Path(event_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._event_log = open(p, "a", encoding="utf-8", buffering=1)
        self._frames: deque[PhorceFeedback] = deque(maxlen=int(max_samples))
        self._latest: PhorceFeedback | None = None
        self._latest_wall: float | None = None
        self._events: deque[_Event] = deque(maxlen=300)
        self._event_seq = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 1kHz 경로
    def push(self, frame: PhorceFeedback) -> None:
        """수신 콜백 전용. 저장만 한다 — 여기에 판단을 넣지 말 것."""
        self._frames.append(frame)      # deque append는 원자적
        self._latest = frame
        # 수신 벽시계 스탬프 — 워치독의 신선도 기준 (로봇 클록과 독립)
        self._latest_wall = time.monotonic()

    # ------------------------------------------------------------ 느린 경로
    def latest(self) -> PhorceFeedback | None:
        return self._latest

    def latest_wall(self) -> float | None:
        """마지막 프레임의 수신 벽시계 시각 (프레임 없으면 None)."""
        return self._latest_wall

    def __len__(self) -> int:
        return len(self._frames)

    def to_arrays(self, window_sec: float | None = None) -> dict:
        """판정용 원본 배열. keys: t, position, velocity, current, dob, temp,
        valid(bool), fault(bool), playing(bool), seq."""
        with self._lock:
            frames = list(self._frames)
        if not frames:
            e2 = np.zeros((0, self.n_axes))
            eb = np.zeros((0, self.n_axes), dtype=bool)
            return {"t": np.zeros(0), "seq": np.zeros(0, dtype=int),
                    "position": e2, "velocity": e2, "current": e2, "dob": e2,
                    "temp": e2, "valid": eb, "fault": eb,
                    "playing": np.zeros(0, dtype=bool)}
        cutoff = frames[-1].t - (self.window_sec if window_sec is None
                                 else float(window_sec))
        frames = [f for f in frames if f.t >= cutoff]
        return {
            "t": np.array([f.t for f in frames]),
            "seq": np.array([f.seq for f in frames], dtype=int),
            "position": np.stack([f.position_rad for f in frames]),
            "velocity": np.stack([f.velocity_rad_s for f in frames]),
            "current": np.stack([f.current_a for f in frames]),
            "dob": np.stack([f.dob_a for f in frames]),
            "temp": np.stack([f.temp_c for f in frames]),
            "valid": np.stack([f.usable for f in frames]),
            "fault": np.stack([f.fault for f in frames]),
            "playing": np.array([f.playing for f in frames], dtype=bool),
        }

    # ------------------------------------------------------------ 이벤트/덤프
    def mark_event(self, text: str, t: float | None = None) -> None:
        if t is None:
            t = self._latest.t if self._latest is not None else 0.0
        self._event_seq += 1
        self._events.append(_Event(t=float(t), text=str(text),
                                   seq=self._event_seq))
        if self._event_log is not None:
            import json
            self._event_log.write(json.dumps(
                {"seq": self._event_seq, "t": float(t),
                 "wall": time.monotonic(), "text": str(text)},
                ensure_ascii=False) + "\n")

    def events(self) -> list:
        return list(self._events)

    def events_since(self, after_seq: int = 0) -> list:
        """seq > after_seq 인 이벤트만 (웹 UI 폴링 커서용)."""
        return [e for e in self._events if e.seq > int(after_seq)]

    def close(self) -> None:
        if self._event_log is not None:
            self._event_log.close()
            self._event_log = None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
        self._latest = None
        self._latest_wall = None
        self._events.clear()

    def dump_text(self, window_sec: float = 2.0) -> str:
        """LLM 프롬프트/사람용 요약. 같은 입력이면 같은 출력."""
        a = self.to_arrays(window_sec)
        rule = "=" * 74
        if len(a["t"]) == 0:
            return f"{rule}\nPHORCE FEEDBACK DUMP | (no frames)\n{rule}"
        t0, t1 = a["t"][0], a["t"][-1]
        lines = [rule,
                 f"PHORCE FEEDBACK DUMP | {len(a['t'])} frames | "
                 f"t = {t0:.3f}..{t1:.3f} s | playing={bool(a['playing'][-1])}",
                 "units: pos [rad]  vel [rad/s]  current/dob [A]  temp [C]",
                 rule,
                 "  ax |  pos_now | vel_now | |cur|max | |dob|max | temp_now | flags"]
        lines.append("  " + "-" * 66)
        for j in range(self.n_axes):
            v = a["valid"][:, j]
            flags = []
            if not v[-1]:
                flags.append("INVALID")
            if a["fault"][-1, j]:
                flags.append("FAULT")
            col = np.where(v, a["current"][:, j], 0.0)
            dob = np.where(v, a["dob"][:, j], 0.0)
            lines.append(
                f"  {j:2d} | {a['position'][-1, j]:8.3f} | {a['velocity'][-1, j]:7.3f} | "
                f"{np.abs(col).max():8.2f} | {np.abs(dob).max():8.2f} | "
                f"{a['temp'][-1, j]:8.1f} | {','.join(flags) or 'ok'}")
        evs = [e for e in self._events if t0 <= e.t <= t1]
        lines.append("[EVENTS]" if evs else "[EVENTS] (none)")
        for e in evs:
            lines.append(f"  t={e.t:.3f}  {e.text}")
        lines.append(rule)
        return "\n".join(lines)
