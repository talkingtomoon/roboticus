"""RealPhorceHAL — phorce 파이썬 파사드 실구현 (공식 문서 5종 확정, 2026-08).

확정된 파사드 계약 (추측 아님 — 문서 대조 완료):
- 연결: phorce.connect()  (기본 target="robot" = 실물). PhorceUnavailable 가능
- 재생: robot.play_async(id, on_feedback=cb) -> PlayHandle
    PlayHandle: .wait(timeout) -> PlayResult, .cancel(), .done(속성),
    .last_feedback
- PlayResult: .ok(★속성 — 괄호 없음), .status_name("SUCCEEDED"|"CANCELED"만
  — "REJECTED"는 존재하지 않는다. 거절은 예외 경로로만 온다),
  .detail(한국어 안내 — 사용자에게 그대로 노출), .completed_count
- 상태: robot.status(timeout=2.0).state_name — "IDLE"만 유휴의 증거.
  STALE/CONTRACT_INACTIVE/UNKNOWN은 "모름" (다른 필드로 idle 추측 금지)
- 카탈로그: robot.motions() 순회 → m.id, m.name
- 진단: robot.doctor() → .ok, .issues
- 예외 계층: PhorceError > PhorceUnavailable | MotionRejected(> MotionBusy)
  | MotionAborted.  MotionRejected: .code(숫자), .reason("REJECT_*"), .detail
- 상수: MIN=1, MAX=50, NO_MOTION=0, MAX_SEQUENCE_LENGTH=1,
  STATE_FRESH_LIMIT_MS=1500 (우리 K_STATE_FRESH_LIMIT_MS와 동일 출처)

거절코드 → 우리 분류는 hal/phorce.py classify_reject_code가 정본이다.
**숫자로 비교한다** — .reason 문자열("REJECT_*") 비교 금지.

피드백은 파사드가 아니라 ROS 2 토픽에서 온다:
adapters/phorce_ros2.py PhorceFeedbackBridge (qos_profile_sensor_data 필수,
시간축은 recv_monotonic_ns). attach_feedback_bridge()로 연결한다.

==========================================================================
스레드 규칙 (어기면 stale이 쌓이거나 BUSY 폭주)
==========================================================================
- 브리지 콜백(1kHz 수신 스레드): _on_frame — 저장 + 팬아웃만. 판단 금지.
- play/play_async: 판단 루프(2Hz) 스레드에서만.
- 핸들 완료 감시는 play마다 전용 데몬 스레드 1개 (파사드 wait 블로킹 격리).
==========================================================================

하지 말 것 (참가자 비공개 — 문서 명시):
- SubmitMotionRequest / /phorce/aperiodic / PhorceCommand 직접 사용 금지
- phorce_monitor 설정 변경 금지
- motion_slot_state 폴링으로 발사 타이밍 재기 금지 (경계 = 핸들 완료,
  폴백 = robot.status().state_name == "IDLE" 한 번)
- CPU 8~11 고정 금지 (로봇 통신 전용 격리 코어)
- play(0) 금지, 한 요청에 id 2개 이상 금지
"""

from __future__ import annotations

import threading

from robot_core.hal.phorce import (
    MotionAborted, MotionBusy, MotionRejected, PhorceError, PhorceFeedback,
    PhorceHAL, PlayHandle, classify_reject_code,
)

_STARTUP_HINT = (
    "시동 절차(README '캠프 시동 절차')를 순서대로 확인:\n"
    "  1) 로봇(pcm·phact) 전원이 먼저 켜져 있는가\n"
    "  2) cat /sys/class/net/eno1/operstate == up 인가\n"
    "  3) 터미널1 phorce_monitor 13개 자가검사 전부 PASS + 켜져 있는가\n"
    "     (로봇을 껐다 켰으면 phorce_monitor도 재실행해야 한다)\n"
    "  4) 터미널2 motion_action_server가 떠 있는가\n"
    "안 되면 운영 담당자 호출. 목 리허설은 MockPhorceHAL로 계속 가능하다.")


def _connect_facade():
    """phorce.connect() — 실패를 조용히 넘기지 않고 시동 절차를 알려준다."""
    try:
        import phorce
    except ImportError as e:
        raise PhorceError(
            "phorce 파사드 모듈을 찾을 수 없다 (import phorce 실패).\n"
            + _STARTUP_HINT) from e
    try:
        return phorce.connect()          # 기본 target="robot" = 실물
    except Exception as e:
        # 파사드 PhorceUnavailable 포함 — 타입 이름으로만 식별한다
        # (파사드 미설치 환경에서도 이 모듈이 import돼야 하므로 isinstance 불가)
        raise PhorceError(
            f"phorce 연결 실패 ({type(e).__name__}: {e}).\n"
            + _STARTUP_HINT) from e


def _translate_exception(exc: Exception) -> PhorceError:
    """파사드 예외 → 우리 예외. 덕 타이핑 — .code가 있으면 거절이다.

    파사드의 MotionRejected/MotionBusy/MotionAborted는 이름이 우리 것과
    같지만 타입은 다르다. 경계에서 전부 우리 타입으로 통일한다.
    """
    code = getattr(exc, "code", None)
    if code is not None:
        code = int(code)
        reason = str(getattr(exc, "reason", "") or "")
        detail = str(getattr(exc, "detail", "") or "")
        if classify_reject_code(code) == "busy":
            return MotionBusy(detail or reason or "code 5")
        return MotionRejected(code, message=reason, detail=detail)
    if isinstance(exc, PhorceError):
        return exc
    return PhorceError(f"{type(exc).__name__}: {exc}")


class RealPhorceHAL(PhorceHAL):
    """phorce 파사드 래퍼 — PhorceHAL 계약 구현.

    사용 (현장):
        hal = RealPhorceHAL()                     # 내부에서 phorce.connect()
        bridge = PhorceFeedbackBridge()           # adapters/phorce_ros2.py
        hal.attach_feedback_bridge(bridge)        # 1kHz 피드백 연결
        sup = Supervisor(hal, ...)

    테스트: robot 인자로 파사드 시그니처를 흉내낸 fake를 주입한다
    (tests/test_real_phorce.py — 파사드 계약 대조가 목적).
    """

    _WAIT_SLICE_S = 5.0     # 파사드 wait를 슬라이스로 끊어 stop 가능하게

    def __init__(self, robot=None, status_timeout_s: float = 2.0) -> None:
        self._robot = robot if robot is not None else _connect_facade()
        self._status_timeout = float(status_timeout_s)
        self._latest: PhorceFeedback | None = None
        self._lock = threading.Lock()
        self._callbacks: list = []
        self._playing = False          # frame.playing 스탬프용 (판단에는 미사용)
        self._bridge = None
        self.frame_cb_errors = 0       # 수신 콜백 예외는 삼키되 계측한다
        self.play_call_count = 0       # BUSY 폭주 계측 (목과 동일 규약)

    # ------------------------------------------------------------ 피드백 경로
    def attach_feedback_bridge(self, bridge) -> None:
        """PhorceFeedbackBridge 연결 — bridge.start(콜백)를 대신 불러준다."""
        self._bridge = bridge
        bridge.start(self._on_frame)

    def _on_frame(self, frame: PhorceFeedback) -> None:
        """브리지 수신 스레드(1kHz)에서 불린다 — 저장 + 팬아웃만.

        frame.playing은 여기서 스탬프한다: 브리지는 재생 여부를 모르고,
        PLAYBACK_STALL 감지가 이 비트를 전제한다.
        """
        frame.playing = self._playing
        with self._lock:
            self._latest = frame
        for cb in self._callbacks:
            try:
                cb(frame)
            except Exception:
                self.frame_cb_errors += 1   # 수신 스레드는 죽으면 안 된다

    def latest_feedback(self) -> PhorceFeedback | None:
        with self._lock:
            return self._latest

    def watch(self, callback) -> None:
        self._callbacks.append(callback)

    # ------------------------------------------------------------ 재생 경로
    def play_async(self, motion_id: int) -> PlayHandle:
        """판단 루프 스레드에서만 부른다. 거절은 여기서 즉시 예외로 온다."""
        self.play_call_count += 1
        motion_id = int(motion_id)
        handle = PlayHandle(motion_id)
        fb = self.latest_feedback()
        handle.t_start = None if fb is None else float(fb.t)
        try:
            facade_handle = self._robot.play_async(motion_id)
        except Exception as e:
            raise _translate_exception(e) from e
        self._playing = True
        threading.Thread(target=self._await_result,
                         args=(facade_handle, handle),
                         daemon=True, name=f"play-wait-{motion_id}").start()
        return handle

    def _await_result(self, facade_handle, handle: PlayHandle) -> None:
        """파사드 핸들 완료 → 우리 핸들 완료/중단. 전용 데몬 스레드.

        PlayResult.status_name은 "SUCCEEDED"|"CANCELED"뿐 — "REJECTED"
        분기는 만들지 않는다 (거절은 play_async에서 예외로 이미 끝났다).
        """
        result = None
        while result is None:
            try:
                result = facade_handle.wait(timeout=self._WAIT_SLICE_S)
            except Exception as e:
                name = type(e).__name__
                # 타임아웃류 + 아직 미완료 → 계속 대기 (긴 모션)
                if "imeout" in name and not getattr(facade_handle, "done", False):
                    continue
                self._finish(handle, aborted=MotionAborted(
                    handle.motion_id,
                    str(getattr(e, "detail", "") or f"{name}: {e}")))
                return
            if result is None and not getattr(facade_handle, "done", False):
                continue                    # wait가 None을 돌려주는 구현 방어
        if getattr(result, "ok", False):    # ★속성 — 괄호 없음 (문서 확정)
            self._finish(handle, aborted=None)
        else:
            status = str(getattr(result, "status_name", "") or "CANCELED")
            detail = str(getattr(result, "detail", "") or "")
            done_n = getattr(result, "completed_count", None)
            reason = status + (f": {detail}" if detail else "") + (
                f" (completed_count={done_n})" if done_n is not None else "")
            self._finish(handle, aborted=MotionAborted(handle.motion_id, reason))

    def _finish(self, handle: PlayHandle, aborted: MotionAborted | None) -> None:
        self._playing = False
        fb = self.latest_feedback()
        handle.t_end = None if fb is None else float(fb.t)
        if aborted is None:
            handle._complete()
        else:
            handle._abort(aborted)

    def play(self, motion_id: int) -> int:
        """블로킹 재생 — play_async 위에서 얻는다."""
        h = self.play_async(motion_id)
        h.wait()
        return h.result()

    # ------------------------------------------------------------ 상태 경로
    def busy_state(self) -> str:
        """robot.status().state_name 기반 폴백. "IDLE"만 유휴의 증거.

        STALE/CONTRACT_INACTIVE/UNKNOWN(및 조회 실패)은 "unknown" —
        감독 루프는 보류한다. 다른 필드로 idle을 추측하지 않는다 (문서 명시).
        """
        try:
            st = self._robot.status(timeout=self._status_timeout)
        except Exception:
            return "unknown"
        name = str(getattr(st, "state_name", "") or "").upper()
        if name == "IDLE":
            return "idle"
        if name in ("STALE", "CONTRACT_INACTIVE", "UNKNOWN", ""):
            return "unknown"
        return "busy"

    def is_busy(self) -> bool:
        """계약 잔존 메서드 — "unknown"은 busy로 치지 않는다.
        판단 경로는 busy_state()를 직접 쓴다 (supervisor가 그렇게 한다)."""
        return self.busy_state() == "busy"

    def status(self) -> dict:
        """preflight용. 파사드 status는 state_name만 확정이므로 그것만 담는다
        (estop/EtherCAT 판정은 doctor()의 몫)."""
        try:
            st = self._robot.status(timeout=self._status_timeout)
            return {"state_name": str(getattr(st, "state_name", "?"))}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def doctor(self) -> dict:
        """robot.doctor() → {"ok": bool, "issues": [str]}. preflight가 본다."""
        d = self._robot.doctor()
        return {"ok": bool(getattr(d, "ok", False)),
                "issues": [str(i) for i in (getattr(d, "issues", None) or [])]}

    # ------------------------------------------------------------ 카탈로그
    def catalog(self) -> dict[int, float]:
        """robot.motions() → {id: 0.0}. 파사드는 duration을 주지 않는다 —
        duration/태그/자세의 정본은 카탈로그 JSON이고, reconcile은 id만
        대조하므로 0.0으로 충분하다."""
        return {int(m.id): 0.0 for m in self._robot.motions()}

    def motion_names(self) -> dict[int, str]:
        """{id: name} — 스모크/로그 표시용 (판단에는 미사용)."""
        return {int(m.id): str(getattr(m, "name", "?"))
                for m in self._robot.motions()}
