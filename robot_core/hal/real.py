"""RealRobotHAL: 실물 로봇용 stub — phact-401 + phorce SDK 전제.

캠프 현장에서 **이 파일만** 채우면 나머지 코드(제어기/로거/테스트)는 그대로 돌아간다.
지금은 시그니처만 있고 전부 NotImplementedError를 던진다.

확정된 하드웨어 (주최측 공개, 2026-08):
  - 엔젤로보틱스 phact-401 6축, FDCAN 통신
  - Jetson AGX (Ubuntu 22.04, aarch64)
  - phorce SDK: C++ / Python / ROS2 바인딩 제공
  - 액추에이터 내장 기능: AFC(액티브 마찰제거), 하드웨어 동작제한
  - 스펙 상수는 robot_core/hal/profiles.py 의 PHACT_401 을 쓸 것 (직접 쓰지 말 것)

--------------------------------------------------------------------------
현장 채우기 순서 — phorce SDK 파이썬 바인딩에서 다음 셋을 찾는 것이 전부다
--------------------------------------------------------------------------
0) SDK 설치 확인 + 문서/예제 확보
   - python -c "import phorce" (실제 모듈명은 SDK 문서 확인)
   - 예제 스크립트가 있으면 그것부터 실행해 통신이 살아있는지 본다.
   - FDCAN 인터페이스 이름/비트레이트는 주최측이 정렬해 놓은 값을 그대로.

1) [매핑 (a)] 상태 읽기 함수 찾기 → read_state()
   - 찾을 것: 관절별 (위치, 속도, 토크)를 돌려주는 함수/콜백.
     후보 이름 패턴: get_state / read_joint_states / feedback / on_state_msg
   - 확인할 것:
     * 단위 (rad? deg? 모터축? 관절축?) — 우리 규약은 관절축 rad, rad/s, Nm
     * 블로킹 여부 — 1kHz 루프에서 블로킹이면 SDK의 주기 콜백 + 캐시 방식으로
     * 타임스탬프 제공 여부 — 있으면 그걸 쓰고, 없으면 time.perf_counter()
     * 토크가 '전류 환산 추정'인지 '출력축 측정'인지 (AFC on이면 마찰이 이미
       상쇄된 값일 수 있다 — 델타 캘리브레이션 해석에 중요. 아래 3) 참고)

2) [매핑 (b)] 토크/임피던스 지령 함수 찾기 → send_command()
   - 찾을 것: (q_des, qd_des, kp, kd, tau_ff)를 받는 임피던스 지령.
     후보 이름 패턴: set_impedance / joint_impedance_command / mit_command
   - 임피던스 모드가 없고 토크 모드만 있으면: tau = kp(q_des-q)+kd(qd_des-qd)+tau_ff
     를 여기서 계산해 토크 지령으로 보낸다 (규약은 인터페이스가 유지).
   - **전송 직전 클램프는 SDK/하드웨어 동작제한과 별개로 반드시 넣는다**:
         tau_ff = clip(tau_ff, ±PHACT_401.tau_limit)   # 연속 5.76 Nm
     하드웨어 제한은 마지막 방어선이고, 우리 클램프가 첫 방어선이다.
   - 게인 단위 확인: SDK가 관절축 기준인지 모터축 기준인지.

3) [매핑 (c)] AFC on/off 상태 조회 찾기 → afc_state 프로퍼티
   - 찾을 것: AFC(액티브 마찰제거) 활성 여부를 돌려주는 함수/설정값.
     후보 이름 패턴: get_afc / friction_compensation / afc_enabled
   - 왜 중요한가: 델타 캘리브레이션이 학습하는 것이 곧 '남아있는 마찰'이다.
     AFC on 상태에서 수집한 모델과 AFC off 상태의 로봇은 서로 안 맞는다 —
     보정 노드가 afc_state 불일치 시 활성화를 거부한다 (delta/corrector_node.py).
   - 조회 API를 못 찾으면 주최측에 현재 설정을 물어보고 아래 프로퍼티에
     상수로라도 박아 둘 것. "unknown"으로 두면 가드가 확인을 못 한다.

4) 안전장치 (estop / disable)
   - SDK의 disable/stop 함수를 찾고, estop()은 게인 0 지령 → disable 순서로.
   - SDK 워치독 유무 확인: 지령이 N ms 끊기면 스스로 멈추는지.

5) 검증 (실제로 움직이기 전에)
   - 모터 1개만 → kp=0, kd=0.5 (댐핑만) → 손으로 돌려 부호 확인.
   - AFC on/off 각각에서 손으로 돌려보고 저항 차이를 체감해 둘 것
     (캘리브레이션 데이터 해석에 도움이 된다).
   - kp 아주 작게(1~2) 홀드 테스트 → 그 다음 나머지 관절.

6) 스모크 테스트
   - tests/test_hal_mock.py 를 RealRobotHAL로, **관절 하나만, kp 낮게,
     진폭 작게** 바꿔 돌릴 것.
   - 그 다음 scripts/field_calibration.py --hal real --budget-min 8
"""

from __future__ import annotations

import numpy as np

from robot_core.hal.interface import JointCommand, JointState, RobotHAL
from robot_core.hal.profiles import PHACT_401


class RealRobotHAL(RobotHAL):
    """phact-401 실물 HAL. 캠프 현장에서 phorce SDK 위에 구현한다.

    Parameters
    ----------
    channel:
        FDCAN 인터페이스 이름. 주최측 정렬값 그대로 (예: "can0").
    motor_ids:
        관절 순서대로 나열한 드라이버 ID. 이 순서가 곧 관절 인덱스다.
        (phorce SDK가 ID 대신 관절 이름/인덱스를 쓰면 그에 맞게 바꿀 것)
    gear_ratio / joint_direction / joint_offset:
        모터축 ↔ 관절축 환산. SDK가 관절축 값을 바로 주면 전부 기본값(1, 1, 0).
    """

    def __init__(
        self,
        channel: str = "can0",
        bitrate: int = 1_000_000,
        motor_ids: list[int] | None = None,
        gear_ratio=1.0,
        joint_direction=1.0,
        joint_offset=0.0,
    ) -> None:
        self.channel = channel
        self.bitrate = bitrate
        self.motor_ids = list(motor_ids) if motor_ids else []
        self.gear_ratio = gear_ratio
        self.joint_direction = joint_direction
        self.joint_offset = joint_offset

        self._sdk = None  # TODO(현장): phorce SDK 핸들 (예: phorce.Robot(...))
        self._connected = False

        # 스펙 상수는 프로파일에서. 관절 한계(각도)는 주최측/데이터시트에서
        # 확인해 채울 것 — 프로파일에는 토크/속도만 있다.
        self._n = len(self.motor_ids)
        self._joint_limits = np.zeros((self._n, 2))          # TODO(현장): 채우기
        self._torque_limits = np.full(self._n, PHACT_401.tau_limit)

    # ------------------------------------------------------------------ HAL
    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def joint_limits(self) -> np.ndarray:
        # TODO(현장): __init__에서 실제 관절 각도 한계를 채운 뒤 이 예외 삭제.
        raise NotImplementedError("RealRobotHAL.joint_limits: 실제 관절 한계를 채울 것")

    @property
    def torque_limits(self) -> np.ndarray:
        # TODO(현장): PHACT_401.tau_limit 로 초기화돼 있다. 주최측 안내가 다르면
        #             profiles.py 를 고치고(출처 주석 갱신) 이 예외를 삭제.
        raise NotImplementedError("RealRobotHAL.torque_limits: 프로파일 값 확인 후 예외 삭제")

    @property
    def afc_state(self) -> str:
        """AFC(액티브 마찰제거) 상태: "on" / "off" / "unknown".

        TODO(현장): 매핑 (c) — phorce SDK의 AFC 조회 API로 교체.
        구현 전 기본값 "unknown"은 예외를 던지지 않는다 (수집기는 돌아가되,
        보정 노드의 AFC 가드가 '확인 불가'로 기록한다).
        """
        return "unknown"

    def read_state(self) -> JointState:
        """phorce SDK 상태 → JointState.  [매핑 (a)]

        TODO(현장):
          1. SDK 상태 함수 호출 (블로킹이면 콜백+캐시 방식으로 전환)
          2. 관절축 (q, qd, tau)로 환산 — 단위/부호/영점 확인:
                 q  = joint_direction * (q_raw / gear_ratio) - joint_offset
          3. 프레임 유실 시: 직전 값 유지 + 카운터 증가 (권장).
             1kHz 루프에서 예외를 던지면 루프가 죽는다.
        """
        raise NotImplementedError("RealRobotHAL.read_state: phorce SDK 상태 함수 매핑 필요")

    def send_command(self, cmd: JointCommand) -> None:
        """JointCommand → phorce SDK 임피던스 지령.  [매핑 (b)]

        TODO(현장):
          1. 안전 클램프 (하드웨어 동작제한과 별개의 첫 방어선):
                 q_des  = np.clip(cmd.q_des, self._joint_limits[:,0], self._joint_limits[:,1])
                 tau_ff = np.clip(cmd.tau_ff, -self._torque_limits, self._torque_limits)
                 kp, kd = np.clip(cmd.kp, 0, KP_MAX), np.clip(cmd.kd, 0, KD_MAX)
          2. SDK 임피던스 지령 호출 (토크 모드만 있으면 여기서 PD 계산)
          3. 전송 실패 시 카운터 증가 + 연속 N회 실패면 estop().
        """
        raise NotImplementedError("RealRobotHAL.send_command: phorce SDK 지령 함수 매핑 필요")

    # ------------------------------------------------------------- 수명주기
    def connect(self) -> None:
        """SDK 초기화 + FDCAN 연결. TODO(현장): phorce SDK init."""
        raise NotImplementedError("RealRobotHAL.connect: phorce SDK 초기화 구현 필요")

    def enable(self) -> None:
        """모터 인에이블. TODO(현장): SDK enable 시퀀스.

        인에이블 직후 첫 지령은 반드시 kp=kd=0, tau_ff=0 이어야 한다.
        """
        raise NotImplementedError("RealRobotHAL.enable: 모터 인에이블 시퀀스 구현 필요")

    def disable(self) -> None:
        """모터 디스에이블. TODO(현장): SDK disable."""
        raise NotImplementedError("RealRobotHAL.disable: 모터 디스에이블 구현 필요")

    def estop(self) -> None:
        """비상 정지: 게인 0 지령 전송 후 disable.

        TODO(현장): 아래 순서를 그대로 구현할 것.
            zero = JointCommand.damping_only(self.n_joints, kd=0.0)
            self.send_command(zero)   # 토크 0
            self.disable()
        """
        raise NotImplementedError("RealRobotHAL.estop: 비상 정지 구현 필요")

    def close(self) -> None:
        """SDK/버스 정리. TODO(현장): SDK shutdown."""
        raise NotImplementedError("RealRobotHAL.close: 종료 처리 구현 필요")

    def __enter__(self) -> "RealRobotHAL":
        self.connect()
        self.enable()
        return self

    def __exit__(self, *exc) -> None:
        # 예외로 빠져나가도 반드시 꺼지도록.
        try:
            self.estop()
        finally:
            self.close()

    # ------------------------------------------------------- 패킹 헬퍼 뼈대
    # phorce SDK가 패킹을 알아서 하면 이 두 함수는 쓸 일이 없다 — 그대로 두되,
    # SDK 우회(원시 FDCAN 프레임)가 필요해질 때만 꺼내 쓸 것.
    @staticmethod
    def _pack_float(value: float, lo: float, hi: float, bits: int) -> int:
        """실수 → 고정소수점 정수 (MIT/Mini-Cheetah 계열 드라이버 공통 패턴)."""
        span = hi - lo
        value = min(max(value, lo), hi)
        return int((value - lo) * ((1 << bits) - 1) / span)

    @staticmethod
    def _unpack_float(raw: int, lo: float, hi: float, bits: int) -> float:
        """고정소수점 정수 → 실수. _pack_float의 역변환."""
        span = hi - lo
        return raw * span / ((1 << bits) - 1) + lo
