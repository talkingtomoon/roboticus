"""하드웨어 프로파일 — 실물 스펙 상수를 한 곳에.

숫자의 출처를 반드시 주석으로 남긴다. 코드 곳곳에 매직넘버로 흩어지면
현장에서 스펙이 정정됐을 때 어디를 고쳐야 하는지 아무도 모르게 된다.

==========================================================================
토크 대역 사용 원칙 (새 노드를 추가할 때 반드시 읽을 것)
==========================================================================
**신호의 지속 시간 특성이 어느 토크 대역을 쓸 수 있는지 결정한다.**

- 과도 신호 (수백 ms 이내: 충격 반응, 전이 스플라인, PD 과도) →
  순간 정격 대역까지 허용 (tau_clamp / tau_veto / tau_detect).
  짧은 버스트는 액추에이터 설계 의도이고, 하드웨어 온도 제한이 최종
  방어선이다. 버스트를 연속 기준으로 자르면 성능만 버린다.

- 지속 신호 (마찰 피드포워드, 홀드 토크, 상시 보정) →
  연속 정격 예산(tau_cont) 안에서만. 상시 주입되는 신호가 버스트 대역을
  침범하면 열 예산을 무기한 초과 주입할 수 있다 — 델타 보정기의 Δτ
  클램프가 tau_clamp가 아니라 tau_cont 기준인 이유.

- 지속 과부하 감시: tau_cont는 순간 상한이 아니라 **1초 이동평균 예산**이다.
  초과 시 CONTINUOUS_OVERLOAD 이벤트로 회복 루프에 알린다 (규칙 폴백:
  동작 속도 하향). 정당한 버스트와 지속 과부하를 구분하는 것이 목적.
==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """액추에이터 스펙 + 안전율 → 용도별 토크 한계 3단 + 속도 한계.

    - tau_clamp  : 순간 정격 × 안전율. 보정기 최종 출력/HAL 전송 직전의
                   하드 클램프. **순간 상한이다** — 지속 신호의 예산이 아니다.
    - tau_detect : 연속 정격 × 1.5. FailureDetector TORQUE_SPIKE 임계.
                   자체 유발 과도 토크(전이 스플라인 등)는 연속 정격 근처까지
                   올라올 수 있으므로, 그 위에 임계를 둬야 자기 동작을
                   충돌로 오검출하지 않는다.
    - tau_veto   : 순간 정격 × 안전율. 스위칭 dream rollout veto 전용 —
                   전이의 짧은 과도 토크는 버스트 대역까지 허용.
    - tau_cont   : 연속 정격 × 안전율. 지속 신호의 예산이자 1초 이동평균
                   감시 기준 (CONTINUOUS_OVERLOAD). 델타 보정 Δτ 클램프,
                   캘리브레이션 수집 중단 기준도 이것.
    - qd_limit   : 최대 속도 × 안전율.
    """

    name: str
    n_joints: int
    tau_continuous: float   # [Nm] 연속 정격
    tau_peak: float         # [Nm] 순간 정격
    qd_max: float           # [rad/s] 최대 속도
    safety_factor: float = 0.8
    detect_factor: float = 1.5   # tau_detect = 연속 정격 × 이 값

    @property
    def tau_clamp(self) -> float:
        """최종 출력 하드 클램프 [Nm] (순간 상한 — 지속 예산 아님)."""
        return self.tau_peak * self.safety_factor

    @property
    def tau_detect(self) -> float:
        """TORQUE_SPIKE 감지 임계 [Nm]."""
        return self.tau_continuous * self.detect_factor

    @property
    def tau_veto(self) -> float:
        """스위칭 dream veto 한계 [Nm]."""
        return self.tau_peak * self.safety_factor

    @property
    def tau_cont(self) -> float:
        """지속 신호 예산 [Nm] — 1초 이동평균 감시 기준."""
        return self.tau_continuous * self.safety_factor

    @property
    def qd_limit(self) -> float:
        return self.qd_max * self.safety_factor

    # ---- 하위 호환 별칭 (기존 코드용 — 새 코드는 tau_cont를 쓸 것) ----
    @property
    def tau_limit(self) -> float:
        return self.tau_cont

    @property
    def tau_limit_peak(self) -> float:
        return self.tau_veto


# 엔젤로보틱스 phact-401 6축.
# 출처: 주최측 하드웨어 공개 자료 (2026-08) —
#   토크 연속 7.2 Nm / 순간 27 Nm, 속도 15.7 rad/s.
#   Jetson AGX (Ubuntu 22.04, aarch64), phorce SDK (C++/Python/ROS2), FDCAN.
#   액추에이터에 AFC(액티브 마찰제거)와 하드웨어 동작제한 내장.
# 산출:
#   tau_clamp  = 27  × 0.8 = 21.6 Nm   (최종 출력 클램프)
#   tau_detect = 7.2 × 1.5 = 10.8 Nm   (충돌 감지 임계)
#   tau_veto   = 27  × 0.8 = 21.6 Nm   (스위칭 veto)
#   tau_cont   = 7.2 × 0.8 = 5.76 Nm   (지속 예산 / 1초 이동평균)
#   qd_limit   = 15.7 × 0.8 = 12.56 rad/s
PHACT_401 = HardwareProfile(
    name="phact-401",
    n_joints=6,
    tau_continuous=7.2,
    tau_peak=27.0,
    qd_max=15.7,
)

PROFILES = {"phact-401": PHACT_401}
