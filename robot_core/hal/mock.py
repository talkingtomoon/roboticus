"""MockRobotHAL: 관절별 독립 1자유도 근사 시뮬레이터.

목적은 "상위 로직 검증"이지 물리 정확도가 아니다.
다관절 커플링(코리올리, 관성 행렬 off-diagonal)은 일부러 무시한다.

관절 i의 운동방정식:
    M_i * qdd = tau_drive - b_i*qd - tau_c_i*sign(qd) - G_i(q) + tau_dist

여기서 tau_drive는 임피던스 지령을 토크로 환산하고 (옵션으로) 포화시킨 값:
    tau_cmd   = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff
    tau_drive = clip(tau_cmd, ±torque_limit)      # enable_saturation=True일 때

시간 진행: send_command()가 호출될 때마다 self.dt 만큼 적분한다.
(제어 루프 러너가 1kHz로 돌면 dt=1e-3으로 맞춰 쓰면 된다.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_core.hal.interface import JointCommand, JointState, RobotHAL


def _as_vec(value, n: int, name: str) -> np.ndarray:
    """스칼라 또는 길이 n 시퀀스를 shape=(n,) float 배열로."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name}: shape {arr.shape}, expected ({n},) or scalar")
    return arr.copy()


@dataclass
class _Disturbance:
    """활성 외란 하나. t_end까지 torque를 관절에 그대로 더한다."""

    joint_idx: int
    torque: float
    t_end: float


class MockRobotHAL(RobotHAL):
    """실물 없이 상위 로직을 돌려보기 위한 목 로봇.

    Parameters
    ----------
    n_joints:
        관절 수.
    dt:
        send_command() 1회당 적분 시간 [s]. 러너 주기와 맞출 것.
    inertia:
        관절별 등가 관성 M [kg·m^2]. 스칼라 또는 길이 n.
    viscous_friction:
        점성 마찰 계수 b [Nm·s/rad]. enable_viscous_friction=False면 무시된다.
    coulomb_friction:
        Coulomb 마찰 크기 tau_c [Nm]. enable_coulomb_friction=False면 무시된다.
    backlash_width:
        백래시 데드존 전체 폭 [rad]. enable_backlash=False면 무시된다.
        모터측 위치가 링크측 위치를 폭/2 만큼 앞서야 링크가 끌려온다.
    gravity_torque:
        중력 토크 진폭 [Nm]. G_i(q) = gravity_torque_i * cos(q_i).
        기본 0 (테스트를 깔끔하게 하려고). 중력 보상 로직을 시험하려면 켤 것.
    torque_limit:
        관절별 최대 토크 [Nm] (대칭).
    joint_limit:
        (n,2) 배열 또는 (min,max) 튜플. 한계에 닿으면 위치를 클램프하고 속도를 0으로 만든다.
    q0:
        초기 위치 [rad].
    """

    def __init__(
        self,
        n_joints: int = 6,
        *,
        dt: float = 1e-3,
        inertia=0.05,
        viscous_friction=0.02,
        coulomb_friction=0.1,
        backlash_width=0.0,
        gravity_torque=0.0,
        torque_limit=30.0,
        joint_limit=(-np.pi, np.pi),
        q0=0.0,
        enable_viscous_friction: bool = True,
        enable_coulomb_friction: bool = True,
        enable_backlash: bool = False,
        enable_saturation: bool = True,
    ) -> None:
        self._n = int(n_joints)
        self.dt = float(dt)

        self.inertia = _as_vec(inertia, self._n, "inertia")
        if np.any(self.inertia <= 0):
            raise ValueError("inertia must be > 0")
        self.viscous_friction = _as_vec(viscous_friction, self._n, "viscous_friction")
        self.coulomb_friction = _as_vec(coulomb_friction, self._n, "coulomb_friction")
        self.backlash_width = _as_vec(backlash_width, self._n, "backlash_width")
        self.gravity_torque = _as_vec(gravity_torque, self._n, "gravity_torque")
        self._torque_limits = _as_vec(torque_limit, self._n, "torque_limit")

        jl = np.asarray(joint_limit, dtype=float)
        if jl.shape == (2,):
            jl = np.tile(jl, (self._n, 1))
        if jl.shape != (self._n, 2):
            raise ValueError(f"joint_limit: shape {jl.shape}, expected ({self._n},2) or (2,)")
        self._joint_limits = jl.copy()

        self.enable_viscous_friction = enable_viscous_friction
        self.enable_coulomb_friction = enable_coulomb_friction
        self.enable_backlash = enable_backlash
        self.enable_saturation = enable_saturation

        # AFC(액티브 마찰제거) 상태. 목은 마찰을 '날것으로' 시뮬레이션하므로
        # 항상 "off" — 실물(phact-401)은 RealRobotHAL.afc_state가 SDK에서 조회.
        # 델타 캘리브레이션의 AFC 가드(수집 시 기록, 로드 시 대조)가 이 값을 쓴다.
        self.afc_state = "off"

        # --- 상태 ---
        self.t = 0.0
        self.q = _as_vec(q0, self._n, "q0")          # 모터측 위치
        self.qd = np.zeros(self._n)
        self._q_link = self.q.copy()                  # 링크측 위치 (백래시 반대편)
        self._qd_link = np.zeros(self._n)
        self._tau_measured = np.zeros(self._n)
        self._last_cmd: JointCommand | None = None

        # --- 고장/외란 주입 ---
        self._disturbances: list[_Disturbance] = []
        self._jammed = np.zeros(self._n, dtype=bool)

        self.step_count = 0

    # ------------------------------------------------------------------ HAL
    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def joint_limits(self) -> np.ndarray:
        return self._joint_limits.copy()

    @property
    def torque_limits(self) -> np.ndarray:
        return self._torque_limits.copy()

    def read_state(self) -> JointState:
        return JointState(
            q=self._q_link.copy(),
            qd=self._qd_link.copy(),
            tau_measured=self._tau_measured.copy(),
            timestamp=self.t,
        )

    def send_command(self, cmd: JointCommand) -> None:
        """지령을 적용하고 dt만큼 시뮬레이션을 진행한다."""
        self._last_cmd = cmd
        self.step(cmd, self.dt)

    # ------------------------------------------------------------- 주입 API
    def inject_disturbance(self, joint_idx: int, torque: float, duration: float) -> None:
        """외란 토크를 duration [s] 동안 관절에 더한다 (충격 감지 로직 테스트용).

        지금 시뮬레이션 시각(self.t)부터 duration 동안 유효하다.
        같은 관절에 여러 번 걸면 합쳐진다.
        """
        self._check_idx(joint_idx)
        if duration <= 0:
            raise ValueError("duration must be > 0")
        self._disturbances.append(
            _Disturbance(joint_idx=int(joint_idx), torque=float(torque), t_end=self.t + float(duration))
        )

    def inject_jam(self, joint_idx: int) -> None:
        """관절이 물리적으로 걸린 상황. 위치가 고정되고 PD 오차가 쌓여 토크가 치솟는다."""
        self._check_idx(joint_idx)
        self._jammed[joint_idx] = True

    def clear_jam(self, joint_idx: int | None = None) -> None:
        """jam 해제. joint_idx=None이면 전부."""
        if joint_idx is None:
            self._jammed[:] = False
        else:
            self._check_idx(joint_idx)
            self._jammed[joint_idx] = False

    def clear_disturbances(self) -> None:
        self._disturbances.clear()

    def is_jammed(self, joint_idx: int) -> bool:
        self._check_idx(joint_idx)
        return bool(self._jammed[joint_idx])

    def active_disturbance(self) -> np.ndarray:
        """현재 유효한 외란 토크 벡터, shape=(n,)."""
        tau = np.zeros(self._n)
        for d in self._disturbances:
            if self.t < d.t_end:
                tau[d.joint_idx] += d.torque
        return tau

    # ---------------------------------------------------------------- 물리
    def step(self, cmd: JointCommand, dt: float | None = None) -> JointState:
        """지령 하나로 dt만큼 전진. send_command()가 내부적으로 쓴다."""
        dt = self.dt if dt is None else float(dt)

        tau_cmd = (
            cmd.kp * (cmd.q_des - self._q_link)
            + cmd.kd * (cmd.qd_des - self._qd_link)
            + cmd.tau_ff
        )
        tau_drive = (
            np.clip(tau_cmd, -self._torque_limits, self._torque_limits)
            if self.enable_saturation
            else tau_cmd
        )

        tau_dist = self.active_disturbance()
        self._expire_disturbances()

        tau_net = tau_drive + tau_dist
        tau_net = tau_net - self.gravity_torque * np.cos(self.q)
        if self.enable_viscous_friction:
            tau_net = tau_net - self.viscous_friction * self.qd
        if self.enable_coulomb_friction:
            tau_net = self._apply_coulomb(tau_net)

        qdd = tau_net / self.inertia

        # jam된 관절: 아무리 밀어도 안 움직인다.
        qdd = np.where(self._jammed, 0.0, qdd)

        # semi-implicit Euler
        self.qd = np.where(self._jammed, 0.0, self.qd + qdd * dt)
        self.q = self.q + self.qd * dt

        self._apply_joint_limits()
        self._update_link_side(dt)

        # 측정 토크 = 관절에 실제로 걸린 구동 토크 + 외란.
        # (충격/접촉이 tau_measured에 그대로 보이게 하려는 의도적 모델링)
        self._tau_measured = tau_drive + tau_dist

        self.t += dt
        self.step_count += 1
        return self.read_state()

    def _apply_coulomb(self, tau_net: np.ndarray) -> np.ndarray:
        """움직이는 관절은 운동 마찰, 거의 멈춘 관절은 정지 마찰(스틱)."""
        tau_c = self.coulomb_friction
        moving = np.abs(self.qd) > 1e-6
        kinetic = tau_net - tau_c * np.sign(self.qd)
        stuck = np.abs(tau_net) <= tau_c
        breakaway = tau_net - tau_c * np.sign(tau_net)
        static = np.where(stuck, 0.0, breakaway)
        return np.where(moving, kinetic, static)

    def _apply_joint_limits(self) -> None:
        lo, hi = self._joint_limits[:, 0], self._joint_limits[:, 1]
        hit_lo = self.q < lo
        hit_hi = self.q > hi
        self.q = np.clip(self.q, lo, hi)
        self.qd = np.where(hit_lo | hit_hi, 0.0, self.qd)

    def _update_link_side(self, dt: float) -> None:
        """백래시 데드존: 모터측 q가 링크측 q_link를 폭/2 넘게 앞서야 링크가 끌려온다."""
        if not self.enable_backlash or not np.any(self.backlash_width > 0):
            self._qd_link = self.qd.copy()
            self._q_link = self.q.copy()
            return

        half = self.backlash_width / 2.0
        prev = self._q_link.copy()
        slack = self.q - self._q_link
        pushed = np.where(slack > half, self.q - half, self._q_link)
        pulled = np.where(slack < -half, self.q + half, pushed)
        self._q_link = pulled
        self._qd_link = (self._q_link - prev) / dt if dt > 0 else np.zeros(self._n)

    def _expire_disturbances(self) -> None:
        if self._disturbances:
            self._disturbances = [d for d in self._disturbances if self.t < d.t_end]

    def _check_idx(self, joint_idx: int) -> None:
        if not 0 <= joint_idx < self._n:
            raise IndexError(f"joint_idx {joint_idx} out of range [0, {self._n})")

    # ---------------------------------------------------------------- 유틸
    def reset(self, q0=None) -> None:
        """상태/주입을 전부 초기화."""
        self.t = 0.0
        self.q = _as_vec(0.0 if q0 is None else q0, self._n, "q0")
        self.qd = np.zeros(self._n)
        self._q_link = self.q.copy()
        self._qd_link = np.zeros(self._n)
        self._tau_measured = np.zeros(self._n)
        self._last_cmd = None
        self._disturbances.clear()
        self._jammed[:] = False
        self.step_count = 0

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의용
        return (
            f"MockRobotHAL(n={self._n}, t={self.t:.3f}s, steps={self.step_count}, "
            f"jammed={np.flatnonzero(self._jammed).tolist()})"
        )
