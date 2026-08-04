"""데이터 수집기 — 여기 궤적을 nominal PD로 추종하며 시계열 기록.

- HAL 인터페이스만 사용: MockRobotHAL이든 RealRobotHAL이든 동일 코드
- 실시간 안전 감시: 토크/속도가 한계의 감시 비율을 넘으면 즉시 중단하되
  그때까지의 데이터는 보존한다 (부분 데이터도 피팅에 쓸 수 있다)
- 수집 품질 리포트: 속도 구간별 샘플 수 + 방향 반전 횟수
  → 현장에서 "데이터 더 모아야 하나?"를 30초 안에 판단
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robot_core.hal.interface import JointCommand, RobotHAL
from robot_core.legacy.delta.excitation import ExcitationPlan


@dataclass
class SafetyLimits:
    """수집 중 감시 임계.

    tau_abs를 주면 그 절대값[Nm]으로 중단 판정한다 — HAL 한계가 순간 정격
    (버스트 대역)일 때 필수다. 캘리브레이션은 저속 탐침이라 연속 예산
    (tau_cont)을 지속적으로 넘으면 그 자체가 이상이므로, "한계의 85%" 같은
    비율 기준은 버스트 한계 기준으로는 너무 느슨하다.
    tau_abs가 None이면 기존처럼 hal.torque_limits × tau_frac.
    """

    tau_frac: float = 0.85
    tau_abs: float | None = None   # [Nm] 절대 중단 임계 (권장: 프로파일 tau_cont)
    qd_abs: float = 4.0        # [rad/s] 절대 속도 상한 (실물 스펙으로 교체)
    consecutive: int = 3        # 연속 N 샘플 초과 시 중단 (단발 노이즈 무시)


@dataclass
class CalibrationData:
    """수집 결과. 배열은 전부 (N, n_joints)."""

    t: np.ndarray
    q_des: np.ndarray
    qd_des: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray
    rate_hz: float
    joints: list[int]
    kp: float
    kd: float
    aborted: bool = False
    abort_reason: str = ""
    # 수집 당시 AFC(액티브 마찰제거) 상태: "on" / "off" / "unknown".
    # 델타 모델이 학습하는 것이 곧 '남아있는 마찰'이므로, AFC 상태가 다른
    # 로봇에 이 데이터로 만든 모델을 올리면 이중/무효 보정이 된다.
    afc_state: str = "unknown"
    # 출처: "queried" = HAL/SDK가 직접 답함, "declared" = 사람이 --afc로 선언,
    # "unknown" = 미확인. 사후 디버깅 때 "이 값이 어디서 왔나"를 남긴다.
    afc_source: str = "unknown"

    @property
    def n_samples(self) -> int:
        return len(self.t)

    @property
    def n_joints(self) -> int:
        return self.q.shape[1]

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, t=self.t, q_des=self.q_des, qd_des=self.qd_des,
            q=self.q, qd=self.qd, tau=self.tau,
            rate_hz=self.rate_hz, joints=np.array(self.joints),
            kp=self.kp, kd=self.kd,
            aborted=self.aborted, abort_reason=np.str_(self.abort_reason),
            afc_state=np.str_(self.afc_state), afc_source=np.str_(self.afc_source),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationData":
        z = np.load(Path(path), allow_pickle=False)
        return cls(
            t=z["t"], q_des=z["q_des"], qd_des=z["qd_des"],
            q=z["q"], qd=z["qd"], tau=z["tau"],
            rate_hz=float(z["rate_hz"]), joints=[int(j) for j in z["joints"]],
            kp=float(z["kp"]), kd=float(z["kd"]),
            aborted=bool(z["aborted"]), abort_reason=str(z["abort_reason"]),
            # 구버전 npz(필드 없음)는 "unknown"
            afc_state=str(z["afc_state"]) if "afc_state" in z.files else "unknown",
            afc_source=str(z["afc_source"]) if "afc_source" in z.files else "unknown",
        )

    # ------------------------------------------------------------ 품질 리포트
    def quality_report(self, speed_bins=(0.02, 0.05, 0.1, 0.3, 0.7, 1.5, 3.0)) -> str:
        lines = ["[COLLECTION QUALITY]"]
        lines.append(f"  samples: {self.n_samples}  ({self.n_samples / self.rate_hz:.1f} s "
                     f"@ {self.rate_hz:.0f} Hz)  AFC={self.afc_state}"
                     + ("  ** ABORTED: " + self.abort_reason if self.aborted else ""))
        edges = [0.0, *speed_bins, np.inf]
        header = " | ".join(f"{edges[i]:>5.2f}-{edges[i + 1]:<5.2f}"
                            for i in range(len(edges) - 1))
        lines.append(f"  speed bins [rad/s]:  {header}")
        for j in self.joints:
            v = np.abs(self.qd[:, j])
            counts = [int(((v >= edges[i]) & (v < edges[i + 1])).sum())
                      for i in range(len(edges) - 1)]
            rev = self.count_reversals(j)
            lines.append(f"  joint {j}:            "
                         + " | ".join(f"{c:>11d}" for c in counts)
                         + f"   reversals={rev}")
        warn = self.warnings()
        if warn:
            lines.append("  경고:")
            lines.extend(f"    - {w}" for w in warn)
        else:
            lines.append("  커버리지 양호 — 피팅 진행 가능")
        return "\n".join(lines)

    def count_reversals(self, joint: int, deadband: float = 0.01) -> int:
        """지령 속도의 부호 반전 횟수 (deadband로 노이즈 무시)."""
        v = self.qd_des[:, joint]
        sign = np.where(v > deadband, 1, np.where(v < -deadband, -1, 0))
        nz = sign[sign != 0]
        return int(np.count_nonzero(np.diff(nz) != 0)) if nz.size >= 2 else 0

    def warnings(self, min_reversals: int = 8, min_slow_samples: int = 300,
                 slow_bin: float = 0.15) -> list[str]:
        out = []
        if self.aborted:
            out.append(f"수집이 중단됨 ({self.abort_reason}) — 부분 데이터만 있음")
        for j in self.joints:
            v = np.abs(self.qd[:, j])
            slow = int(((v > 0.01) & (v < slow_bin)).sum())
            if slow < min_slow_samples:
                out.append(f"joint {j}: 저속 샘플 {slow}개 < {min_slow_samples} "
                           f"(Coulomb 피팅 품질 저하 — 준정적 램프 더 필요)")
            rev = self.count_reversals(j)
            if rev < min_reversals:
                out.append(f"joint {j}: 방향 반전 {rev}회 < {min_reversals} "
                           f"(백래시 데이터 부족 — 반전 세그먼트 더 필요)")
        return out


class Collector:
    """여기 계획을 실행하며 기록한다.

    realtime=False (mock): 최대 속도로 시뮬레이션 스텝 — 8분 계획도 수 초에 리허설.
    realtime=True  (real): 1kHz 벽시계 페이싱.
    """

    def __init__(self, hal: RobotHAL, *, rate_hz: float = 1000.0,
                 kp: float = 40.0, kd: float = 2.0,
                 safety: SafetyLimits | None = None, realtime: bool = False,
                 pause_correction: list | None = None, on_log=None) -> None:
        """pause_correction: 수집 동안 자동으로 꺼둘 보정기 목록 (duck-typed:
        correction_enabled / disable_correction() / enable_correction()).

        보정기가 켜진 채 캘리브레이션을 다시 돌리면 수집 데이터에 보정 토크가
        섞여 이중 보정된다. 수집 시작 시 끄고, 끝나면(중단 포함) 원래 상태로
        복원한다. 원래 꺼져 있던 보정기는 다시 켜지 않는다.
        """
        self.hal = hal
        self.rate_hz = float(rate_hz)
        self.kp = float(kp)
        self.kd = float(kd)
        self.safety = safety or SafetyLimits()
        self.realtime = realtime
        self.pause_correction = list(pause_correction or [])
        self._on_log = on_log

    def collect(self, plan: ExcitationPlan, joints: list[int] | None = None) -> CalibrationData:
        # (B) 보정기 오염 방지: 수집 동안 보정기를 끄고, 끝나면 원상 복원
        was_on = [c for c in self.pause_correction
                  if getattr(c, "correction_enabled", False)]
        for c in was_on:
            c.disable_correction()
            self._log(f"corrector {getattr(c, 'name', '?')} paused for collection")
        try:
            return self._collect_inner(plan, joints)
        finally:
            for c in was_on:
                c.enable_correction()
                self._log(f"corrector {getattr(c, 'name', '?')} restored")

    def _collect_inner(self, plan: ExcitationPlan,
                       joints: list[int] | None = None) -> CalibrationData:
        n = self.hal.n_joints
        dt = 1.0 / self.rate_hz
        n_steps = int(np.ceil(plan.duration * self.rate_hz))
        joints = joints if joints is not None else sorted({w.joint for w in plan.windows})

        if self.safety.tau_abs is not None:
            tau_limit = np.full(n, float(self.safety.tau_abs))
        else:
            tau_limit = self.hal.torque_limits * self.safety.tau_frac
        kp_v = np.full(n, self.kp)
        kd_v = np.full(n, self.kd)

        buf = {k: np.empty((n_steps, n)) for k in ("q_des", "qd_des", "q", "qd", "tau")}
        t_buf = np.empty(n_steps)
        over_count = 0
        aborted, reason = False, ""
        deadline = time.perf_counter()

        k = 0
        for k in range(n_steps):
            t_plan = k * dt
            q_des, qd_des = plan.sample(t_plan)
            self.hal.send_command(JointCommand(
                q_des=q_des, qd_des=qd_des, tau_ff=np.zeros(n), kp=kp_v, kd=kd_v))
            s = self.hal.read_state()

            t_buf[k] = s.timestamp
            buf["q_des"][k] = q_des
            buf["qd_des"][k] = qd_des
            buf["q"][k] = s.q
            buf["qd"][k] = s.qd
            buf["tau"][k] = s.tau_measured

            # --- 실시간 안전 감시 ---
            if (np.any(np.abs(s.tau_measured) > tau_limit)
                    or np.any(np.abs(s.qd) > self.safety.qd_abs)):
                over_count += 1
                if over_count >= self.safety.consecutive:
                    aborted = True
                    j_bad = int(np.argmax(np.maximum(
                        np.abs(s.tau_measured) / np.maximum(tau_limit, 1e-9),
                        np.abs(s.qd) / self.safety.qd_abs)))
                    reason = (f"safety abort at t={t_plan:.2f}s joint {j_bad}: "
                              f"|tau|={abs(s.tau_measured[j_bad]):.2f} "
                              f"(limit {tau_limit[j_bad]:.2f}), "
                              f"|qd|={abs(s.qd[j_bad]):.2f} (limit {self.safety.qd_abs:.2f})")
                    self._log(reason)
                    k += 1
                    break
            else:
                over_count = 0

            if self.realtime:
                deadline += dt
                lag = deadline - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)

        n_kept = k if aborted else n_steps
        data = CalibrationData(
            t=t_buf[:n_kept].copy(),
            q_des=buf["q_des"][:n_kept].copy(), qd_des=buf["qd_des"][:n_kept].copy(),
            q=buf["q"][:n_kept].copy(), qd=buf["qd"][:n_kept].copy(),
            tau=buf["tau"][:n_kept].copy(),
            rate_hz=self.rate_hz, joints=list(joints), kp=self.kp, kd=self.kd,
            aborted=aborted, abort_reason=reason,
            afc_state=str(getattr(self.hal, "afc_state", "unknown")),
            afc_source=("queried"
                        if str(getattr(self.hal, "afc_state", "unknown")) in ("on", "off")
                        else "unknown"),
        )
        self._log(f"collected {data.n_samples} samples"
                  + (" (ABORTED, partial data kept)" if aborted else ""))
        return data

    def _log(self, msg: str) -> None:
        if self._on_log is not None:
            try:
                self._on_log(msg)
            except Exception:
                pass
