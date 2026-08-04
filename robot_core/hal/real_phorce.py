"""RealPhorceHAL — phorce 파사드를 PhorceHAL 계약으로 감싸는 골격.

파사드 API의 정확한 이름은 아직 모른다. 하지만 **매핑 지점 5개**와
**변환 규칙**(거절코드→예외, 스레드 규칙)은 지금 박아둘 수 있다 —
캠프 1일차에 scripts/field_smoke.py가 각 가정(A1~A6)을 실측하고,
깨진 가정의 지점만 여기서 고치면 된다.

==========================================================================
스레드 규칙 (매뉴얼 — 어기면 stale이 쌓이거나 BUSY 폭주가 난다)
==========================================================================
- watch() 콜백(1kHz, 파사드 수신 스레드): **저장만**. 판단·전송·락 대기 금지.
  우리 쪽에서는 FeedbackCache.push만 연결한다 (Supervisor가 해줌).
- play/play_async: **판단 루프(2Hz) 스레드에서만** 부른다.
  콜백/모니터 스레드에서 부르면 안 된다.
==========================================================================
"""

from __future__ import annotations

from robot_core.hal.phorce import (
    MotionBusy, MotionRejected, PhorceError, PhorceFeedback, PhorceHAL,
    PlayHandle, REJECT_BUSY, REJECT_NOT_LOADED, REJECT_NOT_READY,
    REJECT_OUT_OF_RANGE, REJECT_RECOVERY_REQUIRED,
)

# ------------------------------------------------------------ 거절코드 변환표
# 파사드가 예외 대신 코드를 돌려주는 경우 이 표로 예외를 만든다.
# (파사드가 자체 예외 MotionBusy/MotionRejected/MotionAborted를 던지면
#  그대로 우리 예외로 재포장 — 이름이 같아도 타입은 우리 것으로 통일)
REJECT_CODE_MAP = {
    REJECT_BUSY: "busy",                    # 5  → MotionBusy (재시도는 다음 틱)
    REJECT_NOT_READY: "operator",           # 12 → needs_operator=True
    REJECT_RECOVERY_REQUIRED: "operator",   # 13 → needs_operator=True
    REJECT_OUT_OF_RANGE: "permanent",       # 3  → 영구 제외 (supervisor가 함)
    REJECT_NOT_LOADED: "permanent",         # 4  → 영구 제외
}


def raise_for_code(code: int, message: str = "") -> None:
    """거절코드 → 예외. 표에 없는 코드도 조용히 넘기지 않는다."""
    kind = REJECT_CODE_MAP.get(int(code))
    if kind == "busy":
        raise MotionBusy(message or "code 5")
    # operator/permanent/미지 코드 전부 MotionRejected —
    # needs_operator 판정은 MotionRejected가 코드로 직접 한다 (12/13)
    raise MotionRejected(int(code), message)


class RealPhorceHAL(PhorceHAL):
    """phorce 파사드 래퍼. 매핑 지점 5개(M1~M5)를 채우면 완성.

    사용 (현장):
        import phorce
        hal = RealPhorceHAL(phorce.connect())
    """

    def __init__(self, robot=None) -> None:
        if robot is None:
            try:
                import phorce                      # 파사드 모듈 (이름 확인!)
            except ImportError as e:
                # 조용한 실패 금지 — 뭘 해야 하는지까지 말한다
                raise PhorceError(
                    "phorce 파사드를 찾을 수 없다. 젯슨에서:\n"
                    "  1) `phorce doctor` 실행해 설치/버스 상태 확인\n"
                    "  2) 파이썬 바인딩 모듈명이 'phorce'가 맞는지 확인\n"
                    "  3) 안 되면 운영 담당자 호출\n"
                    "목 리허설은 MockPhorceHAL로 계속 가능하다.") from e
            robot = phorce.connect()
        self._robot = robot
        self._last_frame: PhorceFeedback | None = None
        self._callbacks: list = []

    # ---------------------------------------------------------------- M1
    def latest_feedback(self) -> PhorceFeedback | None:
        """M1: 최신 피드백. [검증: field_smoke A1(왕복), A5(valid 축)]

        TODO(현장):
          - 파사드 최신상태 API(예: self._robot.state())를 PhorceFeedback으로 변환
          - 12축 배열 필드명 대조: position_rad/velocity_rad_s/current_a/
            dob_a/bus_v/temp_c/kp_echo/kd_echo + valid/oper/stale/fault
          - 파사드가 최신 캐시를 안 주면: watch 콜백에서 self._last_frame 갱신
            (콜백은 저장만! 변환이 무거우면 원시 msg를 저장하고 여기서 변환)
        """
        raise NotImplementedError("M1: 파사드 상태 → PhorceFeedback 변환 매핑")

    # ---------------------------------------------------------------- M2
    def watch(self, callback) -> None:
        """M2: 1kHz 콜백 등록. [검증: field_smoke A2(주파수/호출 스레드)]

        TODO(현장):
          - 파사드 콜백 등록 API(예: self._robot.watch(fn))에 연결
          - A2가 "메인 스레드에서 불림"으로 나오면 가정이 틀린 것 —
            수신 전용 스레드를 우리가 만들어야 한다 (커밋 메시지에 기록)
          - 콜백 안에서는 변환+callback(frame)만. 예외는 삼킨다 (수신 스레드
            보호) — 단 카운터로 계측할 것.
        """
        raise NotImplementedError("M2: 파사드 watch 등록 매핑")

    # ---------------------------------------------------------------- M3
    def play_async(self, motion_id: int) -> PlayHandle:
        """M3: 비동기 재생. [검증: field_smoke A3(핸들 지연), A4(BUSY)]

        TODO(현장):
          - 파사드 play_async(예: self._robot.play_async(id))가 주는 완료
            신호(future/콜백/이벤트)를 PlayHandle로 변환:
              handle = PlayHandle(motion_id)
              파사드 완료 → handle._complete()
              파사드 MotionAborted → handle._abort(MotionAborted(id, 사유))
          - 거절: 파사드가 코드를 주면 raise_for_code(code),
            예외를 던지면 우리 타입으로 재포장
          - **판단 루프 스레드에서만 불린다** (스레드 규칙)
        """
        raise NotImplementedError("M3: 파사드 play_async → PlayHandle 매핑")

    def play(self, motion_id: int) -> int:
        """블로킹 재생 — M3 위에서 공짜로 얻는다 (별도 매핑 불필요)."""
        h = self.play_async(motion_id)
        h.wait()
        return h.result()

    # ---------------------------------------------------------------- M4
    def is_busy(self) -> bool:
        """M4: 재생 중 여부. [검증: field_smoke A4]

        TODO(현장):
          - 파사드 상태(예: self._robot.status()["playing"])에서.
          - 핸들 유실 폴백 경계 감지에 쓰인다 — 1차 수단은 어디까지나
            M3의 핸들 완료 이벤트다.
        """
        raise NotImplementedError("M4: 파사드 busy 상태 매핑")

    # ---------------------------------------------------------------- M5
    def catalog(self) -> dict[int, float]:
        """M5: 적재 슬롯 정본. [검증: field_smoke A1]

        TODO(현장):
          - robot.motions() 반환 형식을 {id: duration_s}로 변환.
            duration을 안 주면 값은 0.0으로 두고 카탈로그 JSON의 duration을
            정본으로 쓴다 (MotionCatalog.reconcile는 id만 대조).
        """
        raise NotImplementedError("M5: robot.motions() → {id: duration} 매핑")

    def status(self) -> dict:
        """preflight용 상태. [검증: preflight — E-stop/EtherCAT]

        TODO(현장):
          - 최소 키: estop_active(bool), ethercat_operational(bool).
            파사드 status()의 실제 키 이름을 대조해 매핑.
        """
        raise NotImplementedError("status: estop_active/ethercat_operational 매핑")
