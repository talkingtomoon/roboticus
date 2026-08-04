"""phorce 인터페이스 HAL — "관찰 → 판단 → 선곡" 아키텍처의 하드웨어 경계.

실제 제약 (phorce 인터페이스 문서, 2026-08 확정):
- 하행: play(motion_id) 하나뿐. id는 1~50 정수. 궤적/토크/게인 전송 경로 없음
  (저수준 명령은 의도적 비공개 — 우회 시도 금지)
- 큐 없음: 재생 중 play()는 BUSY(5) 거절. 재생 중 중단/개입 불가
- 상행: 1kHz 피드백, 12축 (position/velocity/current/dob/bus_v/temp/kp_echo/
  kd_echo + valid/oper/stale/fault 비트)
- 거절 코드: 3=범위 밖, 4=미적재, 5=BUSY(재시도 가능),
  12=NOT_READY(사람이 영점), 13=RECOVERY_REQUIRED(사람이 복구)
- 모션은 행사 전 phorce Studio 교시로 SD카드에 저장 — 코드로 만들 수 없다

핵심 규칙: 1kHz 콜백에서는 최신 상태 저장만. 판단·전송은 느린 루프(~2Hz).
이 규칙은 supervisor.FeedbackCache / DecisionLoop 구조가 강제한다.

실물 연결(캠프 현장): phorce 파이썬 파사드(phorce.connect())를 이 인터페이스로
감싸는 RealPhorceHAL을 작성한다 — scripts/field_smoke.py가 파사드 가정을
하나씩 검증해 주므로, 깨지는 가정만 고치면 된다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from robot_core.hal.interface import JointCommand
from robot_core.hal.mock import MockRobotHAL

N_AXES = 12
MOTION_ID_MIN, MOTION_ID_MAX = 1, 50

# 거절 코드 (매뉴얼 정본)
REJECT_OUT_OF_RANGE = 3
REJECT_NOT_LOADED = 4
REJECT_BUSY = 5
REJECT_NOT_READY = 12        # 사람이 영점 버튼을 눌러야 함
REJECT_RECOVERY_REQUIRED = 13  # 사람이 복구 절차를 밟아야 함
OPERATOR_CODES = (REJECT_NOT_READY, REJECT_RECOVERY_REQUIRED)


# ------------------------------------------------------------------ 예외 계층
class PhorceError(Exception):
    """phorce 경계에서 나오는 모든 예외의 루트."""


class MotionBusy(PhorceError):
    """코드 5: 재생 중. 유일하게 '나중에 재시도'가 올바른 대응인 거절.

    단, 느린 루프 규칙상 재시도는 다음 판단 틱에서 — 즉시 재시도 루프 금지
    (BUSY 폭주). 애초에 is_busy()/핸들 완료 확인 후에만 play해야 한다.
    """

    code = REJECT_BUSY


class MotionRejected(PhorceError):
    """코드 3/4/12/13: 재시도해도 소용없는 거절.

    needs_operator=True (12/13)면 사람 개입 전까지 어떤 play도 성공하지 않는다
    — 자동 재시도 금지, WAITING_OPERATOR 상태로 전환하고 콘솔 경고.
    """

    def __init__(self, code: int, message: str = "") -> None:
        self.code = int(code)
        self.needs_operator = self.code in OPERATOR_CODES
        super().__init__(f"rejected code={self.code}"
                         + (f" ({message})" if message else "")
                         + (" [사람 개입 필요]" if self.needs_operator else ""))


class MotionAborted(PhorceError):
    """재생이 완주하지 못하고 중단됨 (하드웨어 측 사유). 로봇은 임의 자세에 있다."""

    def __init__(self, motion_id: int, reason: str = "") -> None:
        self.motion_id = int(motion_id)
        self.reason = reason
        super().__init__(f"motion {motion_id} aborted"
                         + (f": {reason}" if reason else ""))


# ------------------------------------------------------------------ 피드백
@dataclass
class PhorceFeedback:
    """1kHz 피드백 프레임 하나. 배열은 전부 shape=(12,).

    valid=False(또는 stale=True)인 축의 수치는 신뢰하지 말 것 —
    감지기는 그 축을 판정에서 제외한다.
    """

    t: float
    seq: int
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    current_a: np.ndarray
    dob_a: np.ndarray          # 외란 관측기 추정 [A 환산] — 충격/접촉 감지의 1차 신호
    bus_v: np.ndarray
    temp_c: np.ndarray
    kp_echo: np.ndarray
    kd_echo: np.ndarray
    valid: np.ndarray          # bool (12,)
    oper: np.ndarray           # bool (12,)
    stale: np.ndarray          # bool (12,)
    fault: np.ndarray          # bool (12,)
    axis_valid_mask: int = 0
    status_flags: int = 0
    playing: bool = False      # 프레임 시점에 재생 중이었나 (수신측 기록)

    @property
    def usable(self) -> np.ndarray:
        """판정에 써도 되는 축 마스크: valid & ~stale."""
        return self.valid & ~self.stale


class PlayHandle:
    """play_async 결과 핸들. 완료 이벤트가 경계 감지의 1차 수단이다
    (is_busy 폴링은 핸들 유실 시 폴백)."""

    def __init__(self, motion_id: int) -> None:
        self.motion_id = int(motion_id)
        self._done = threading.Event()
        self._aborted: MotionAborted | None = None
        self.t_start: float | None = None
        self.t_end: float | None = None

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def result(self) -> int:
        """완료면 motion_id 반환, 중단이면 MotionAborted를 던진다."""
        if not self._done.is_set():
            raise RuntimeError("motion still playing — done()을 먼저 확인할 것")
        if self._aborted is not None:
            raise self._aborted
        return self.motion_id

    # 내부용
    def _complete(self) -> None:
        self._done.set()

    def _abort(self, exc: MotionAborted) -> None:
        self._aborted = exc
        self._done.set()


# ------------------------------------------------------------------ 인터페이스
class PhorceHAL:
    """phorce 파사드의 우리 쪽 경계. Mock/Real이 이 계약을 지킨다.

    - latest_feedback(): 논블로킹, 마지막 수신 프레임 (없으면 None)
    - play(id): 블로킹 재생. MotionBusy/MotionRejected/MotionAborted
    - play_async(id): PlayHandle 반환 (완료 이벤트 = 경계 감지 1차 수단)
    - is_busy(): 재생 중인가
    - catalog(): 적재 슬롯 정본 {id: duration_s} (robot.motions() 대응)
    """

    def latest_feedback(self) -> PhorceFeedback | None:
        raise NotImplementedError

    def play(self, motion_id: int) -> int:
        raise NotImplementedError

    def play_async(self, motion_id: int) -> PlayHandle:
        raise NotImplementedError

    def is_busy(self) -> bool:
        raise NotImplementedError

    def catalog(self) -> dict[int, float]:
        raise NotImplementedError

    def watch(self, callback) -> None:
        """1kHz 피드백 콜백 등록. **콜백에서는 최신 상태 저장만 할 것** —
        판단/전송을 여기 넣으면 수신 스레드가 밀려 stale이 쌓인다."""
        raise NotImplementedError


# ------------------------------------------------------------------ 목 구현
@dataclass
class MockMotion:
    """목 카탈로그 항목: (지속시간, 궤적생성함수 t→q_des(12,))."""

    duration_s: float
    traj_fn: object            # callable (t: float) -> np.ndarray (12,)


class MockPhorceHAL(PhorceHAL):
    """재생 시뮬레이터. MockRobotHAL의 관절 동역학을 재사용해 재생 중
    피드백 스트림을 생성한다.

    - 시간은 step(n_steps)로만 흐른다 (결정적 리허설)
    - play()는 블로킹: 내부에서 완주까지 step
    - 외란 주입 유지: inject_disturbance()가 dob_a 필드로 그대로 노출된다
      (실물 DOB가 추정하는 외란 토크의 ground truth)
    - 12/13 거절 주입: set_rejection(code) — clear_rejection()까지 지속
      ("사람이 버튼을 누를 때까지"의 모사)
    - abort_playback(): 재생 중단 → 핸들이 MotionAborted. 로봇은 그 자리(임의
      자세)에 남는다 — 선택기의 진입 하드 필터가 처리해야 하는 상황
    """

    # 내부 위치 제어기 게인 (실물의 내장 제어기 모사 — 우리가 못 바꾸는 값)
    _KP, _KD = 60.0, 2.5
    _CURRENT_PER_NM = 1.0      # 토크→전류 환산 [A/Nm] (목 단순화: 1)

    def __init__(
        self,
        catalog: dict[int, MockMotion | tuple],
        *,
        dt: float = 1e-3,
        loaded_ids: set[int] | None = None,
        q0: np.ndarray | None = None,
        temp_ambient: float = 35.0,
        heat_k: float = 0.35,      # dT/dt += heat_k * current^2
        cool_k: float = 0.08,      # dT/dt -= cool_k * (T - ambient)
        dynamics_kwargs: dict | None = None,
    ) -> None:
        self._catalog: dict[int, MockMotion] = {}
        for mid, m in catalog.items():
            if not (MOTION_ID_MIN <= int(mid) <= MOTION_ID_MAX):
                raise ValueError(f"motion id {mid}: 1~50 범위 밖")
            self._catalog[int(mid)] = m if isinstance(m, MockMotion) else MockMotion(*m)
        self._loaded = set(loaded_ids) if loaded_ids is not None else set(self._catalog)

        dk = {"n_joints": N_AXES, "dt": dt, "torque_limit": 21.6,
              "coulomb_friction": 0.12, "viscous_friction": 0.045}
        dk.update(dynamics_kwargs or {})
        self._dyn = MockRobotHAL(**dk)
        if q0 is not None:
            self._dyn.q[:] = np.asarray(q0, dtype=float)
            self._dyn._q_link[:] = self._dyn.q
        self.dt = float(dt)

        self._playing: int | None = None
        self._play_t0 = 0.0
        self._handle: PlayHandle | None = None
        self._hold_q = self._dyn.read_state().q.copy()

        self._rejection_code: int | None = None
        self._seq = 0
        self._latest: PhorceFeedback | None = None
        self._callbacks: list = []
        self._lock = threading.Lock()

        self.temp_c = np.full(N_AXES, float(temp_ambient))
        self._temp_ambient = float(temp_ambient)
        self._heat_k, self._cool_k = float(heat_k), float(cool_k)
        self._valid = np.ones(N_AXES, dtype=bool)
        self._stale = np.zeros(N_AXES, dtype=bool)
        self._fault = np.zeros(N_AXES, dtype=bool)
        self.play_call_count = 0    # BUSY 폭주 테스트용 계측

        self.step(1)  # 첫 프레임 생성 (latest_feedback가 바로 쓸 수 있게)

    # ------------------------------------------------------------ 주입 API
    def inject_disturbance(self, axis: int, torque: float, duration: float) -> None:
        self._dyn.inject_disturbance(axis, torque, duration)

    def inject_jam(self, axis: int) -> None:
        """축이 물체에 막혀 진행 불가 (재생은 시간 기준으로 계속 흐른다 —
        실물도 그렇다. PLAYBACK_STALL 감지 시나리오용)."""
        self._dyn.inject_jam(axis)

    def clear_jam(self, axis: int | None = None) -> None:
        self._dyn.clear_jam(axis)

    def set_rejection(self, code: int) -> None:
        """다음 play부터 이 코드로 거절 (clear_rejection까지 지속)."""
        if code not in (REJECT_NOT_READY, REJECT_RECOVERY_REQUIRED,
                        REJECT_OUT_OF_RANGE, REJECT_NOT_LOADED):
            raise ValueError(f"주입 가능한 거절 코드가 아님: {code}")
        self._rejection_code = int(code)

    def clear_rejection(self) -> None:
        self._rejection_code = None

    def set_temperature(self, axis: int, temp_c: float) -> None:
        self.temp_c[axis] = float(temp_c)

    def set_axis_valid(self, axis: int, valid: bool) -> None:
        self._valid[axis] = bool(valid)

    def inject_fault(self, axis: int) -> None:
        self._fault[axis] = True

    def clear_fault(self, axis: int) -> None:
        self._fault[axis] = False

    def abort_playback(self, reason: str = "injected abort") -> None:
        """재생 강제 중단 — 로봇은 현재(임의) 자세에 남는다."""
        if self._playing is None:
            return
        mid = self._playing
        self._playing = None
        self._hold_q = self._dyn.read_state().q.copy()
        if self._handle is not None:
            self._handle._abort(MotionAborted(mid, reason))
            self._handle = None

    # ------------------------------------------------------------- 인터페이스
    def latest_feedback(self) -> PhorceFeedback | None:
        with self._lock:
            return self._latest

    def is_busy(self) -> bool:
        return self._playing is not None

    def catalog(self) -> dict[int, float]:
        """적재 슬롯 정본 (robot.motions() 대응): {id: duration_s}."""
        return {mid: self._catalog[mid].duration_s for mid in sorted(self._loaded)}

    def watch(self, callback) -> None:
        self._callbacks.append(callback)

    def play_async(self, motion_id: int) -> PlayHandle:
        self.play_call_count += 1
        motion_id = int(motion_id)
        if self._playing is not None:
            raise MotionBusy(f"motion {self._playing} still playing")
        if self._rejection_code is not None:
            raise MotionRejected(self._rejection_code)
        if not (MOTION_ID_MIN <= motion_id <= MOTION_ID_MAX):
            raise MotionRejected(REJECT_OUT_OF_RANGE, f"id {motion_id}")
        if motion_id not in self._loaded or motion_id not in self._catalog:
            raise MotionRejected(REJECT_NOT_LOADED, f"id {motion_id}")

        self._playing = motion_id
        self._play_t0 = self._dyn.t
        handle = PlayHandle(motion_id)
        handle.t_start = self._dyn.t
        self._handle = handle
        return handle

    def play(self, motion_id: int) -> int:
        """블로킹 재생: 완주(또는 중단)까지 시뮬레이션을 돌린다."""
        handle = self.play_async(motion_id)
        guard = 0
        max_steps = int(self._catalog[motion_id].duration_s / self.dt) + 10
        while not handle.done() and guard <= max_steps:
            self.step(1)
            guard += 1
        return handle.result()

    # ---------------------------------------------------------------- 시뮬
    def step(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            self._step_once()

    def _step_once(self) -> None:
        # 재생 중이면 궤적 추종, 아니면 마지막 자세 홀드 (실물의 내장 제어기)
        if self._playing is not None:
            m = self._catalog[self._playing]
            t_rel = self._dyn.t - self._play_t0
            if t_rel >= m.duration_s:
                self._finish_playback()
                q_des = self._hold_q
            else:
                q_des = np.asarray(m.traj_fn(t_rel), dtype=float)
        else:
            q_des = self._hold_q

        n = N_AXES
        self._dyn.send_command(JointCommand(
            q_des=q_des, qd_des=np.zeros(n), tau_ff=np.zeros(n),
            kp=np.full(n, self._KP), kd=np.full(n, self._KD)))
        state = self._dyn.read_state()

        current = state.tau_measured * self._CURRENT_PER_NM
        # 온도: 1차 열 모델 (전류^2 발열 - 냉각)
        self.temp_c += (self._heat_k * current**2
                        - self._cool_k * (self.temp_c - self._temp_ambient)) * self.dt

        self._seq += 1
        frame = PhorceFeedback(
            t=state.timestamp, seq=self._seq,
            position_rad=state.q.copy(),
            velocity_rad_s=state.qd.copy(),
            current_a=current.copy(),
            # DOB: 외란 관측기의 이상적 목 — 주입된 외란 토크가 그대로 보인다
            dob_a=self._dyn.active_disturbance() * self._CURRENT_PER_NM,
            bus_v=np.full(N_AXES, 48.0),
            temp_c=self.temp_c.copy(),
            kp_echo=np.full(N_AXES, self._KP),
            kd_echo=np.full(N_AXES, self._KD),
            valid=self._valid.copy(),
            oper=np.ones(N_AXES, dtype=bool),
            stale=self._stale.copy(),
            fault=self._fault.copy(),
            axis_valid_mask=int(np.packbits(
                np.pad(self._valid, (0, 4)), bitorder="little")[:2].view(np.uint16)[0]),
            status_flags=1 if self._playing is not None else 0,
            playing=self._playing is not None,
        )
        with self._lock:
            self._latest = frame
        for cb in self._callbacks:
            cb(frame)   # 실물 규칙과 동일: 콜백은 저장만 해야 한다

    def _finish_playback(self) -> None:
        mid = self._playing
        m = self._catalog[mid]
        self._playing = None
        self._hold_q = np.asarray(m.traj_fn(m.duration_s), dtype=float)
        if self._handle is not None:
            self._handle.t_end = self._dyn.t
            self._handle._complete()
            self._handle = None

    @property
    def t(self) -> float:
        return self._dyn.t
