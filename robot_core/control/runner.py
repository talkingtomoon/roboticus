"""제어 루프 러너.

목표 주파수(기본 1kHz)로 다음을 반복한다:
    state = hal.read_state()
    cmd   = policy(state)
    hal.send_command(cmd)

실제 달성 주파수와 지터를 재서 LoopStats로 돌려준다.
정확도가 아니라 "우리 루프가 실제로 몇 Hz로 도는지"를 아는 게 목적이다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from robot_core.hal.interface import JointCommand, JointState, RobotHAL

Policy = Callable[[JointState], JointCommand]


@dataclass
class LoopStats:
    """한 번의 run()에 대한 타이밍 통계. 단위는 dt_* 가 [ms], 나머지는 표기대로."""

    target_hz: float
    n_steps: int = 0
    duration_sec: float = 0.0
    achieved_hz: float = 0.0
    dt_mean_ms: float = 0.0
    dt_std_ms: float = 0.0      # 지터 (루프 주기 표준편차)
    dt_min_ms: float = 0.0
    dt_max_ms: float = 0.0
    dt_p50_ms: float = 0.0
    dt_p95_ms: float = 0.0
    dt_p99_ms: float = 0.0
    compute_mean_ms: float = 0.0  # read_state+policy+send_command 소요
    compute_p50_ms: float = 0.0
    compute_p95_ms: float = 0.0
    compute_max_ms: float = 0.0   # OS 선점 한 번에 크게 튄다. 진단은 p95를 볼 것
    overruns: int = 0             # 주기 안에 일을 못 끝낸 횟수
    policy_errors: int = 0
    periods_ms: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    @property
    def overrun_rate(self) -> float:
        return self.overruns / self.n_steps if self.n_steps else 0.0

    def summary(self) -> str:
        return (
            f"loop: target {self.target_hz:.0f} Hz -> achieved {self.achieved_hz:.1f} Hz "
            f"({self.n_steps} steps in {self.duration_sec:.3f}s)\n"
            f"  period  mean {self.dt_mean_ms:.3f} ms | jitter(std) {self.dt_std_ms:.3f} ms | "
            f"min {self.dt_min_ms:.3f} | p50 {self.dt_p50_ms:.3f} | "
            f"p95 {self.dt_p95_ms:.3f} | p99 {self.dt_p99_ms:.3f} | max {self.dt_max_ms:.3f}\n"
            f"  compute mean {self.compute_mean_ms:.3f} ms | p50 {self.compute_p50_ms:.3f} | "
            f"p95 {self.compute_p95_ms:.3f} | max {self.compute_max_ms:.3f} ms\n"
            f"  overruns {self.overruns}/{self.n_steps} ({100 * self.overrun_rate:.2f}%) | "
            f"policy errors {self.policy_errors}"
        )


class ControlLoopRunner:
    """고정 주기 제어 루프.

    Parameters
    ----------
    hal:
        RobotHAL 구현체 (목이든 실물이든).
    rate_hz:
        목표 주파수. 기본 1000.
    logger:
        RingLogger (선택). 있으면 매 스텝 기록한다.
    spin_wait:
        True면 마감 직전 구간을 busy-wait으로 채워 정밀도를 올린다 (CPU 1코어 점유).
        Windows에서 time.sleep 해상도가 나빠서 1kHz에는 사실상 필수.
    on_error:
        policy가 예외를 던졌을 때 동작. "raise"(기본)면 안전 지령을 보내고 예외를 올린다.
        "hold"면 카운트만 올리고 직전 지령을 유지한 채 루프를 계속한다.
    """

    def __init__(
        self,
        hal: RobotHAL,
        rate_hz: float = 1000.0,
        logger=None,
        *,
        spin_wait: bool = True,
        on_error: str = "raise",
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        if on_error not in ("raise", "hold"):
            raise ValueError('on_error must be "raise" or "hold"')

        self.hal = hal
        self.rate_hz = float(rate_hz)
        self.period = 1.0 / self.rate_hz
        self.logger = logger
        self.spin_wait = spin_wait
        self.on_error = on_error

        self._policy: Policy | None = None
        self._stop = False
        self.last_stats: LoopStats | None = None
        # 로거에 넘길 벽시계 기준점. run()을 여러 번 나눠 호출해도 시간이
        # 뒤로 가지 않도록 러너 생성 시점에 한 번만 잡는다.
        self._t_origin = time.perf_counter()

    def set_policy(self, fn: Policy) -> None:
        """fn(state) -> JointCommand."""
        if not callable(fn):
            raise TypeError("policy must be callable: fn(state) -> JointCommand")
        self._policy = fn

    def stop(self) -> None:
        """다음 스텝 경계에서 루프를 멈춘다 (policy 안에서 호출 가능)."""
        self._stop = True

    def run(self, duration_sec: float | None = None, n_steps: int | None = None) -> LoopStats:
        """루프를 돌린다. duration_sec 또는 n_steps 중 하나는 지정해야 한다."""
        if self._policy is None:
            raise RuntimeError("policy not set — runner.set_policy(fn) 먼저 호출할 것")
        if duration_sec is None and n_steps is None:
            raise ValueError("duration_sec 또는 n_steps 중 하나는 지정할 것")

        max_steps = n_steps if n_steps is not None else int(np.ceil(duration_sec * self.rate_hz))
        self._stop = False

        periods: list[float] = []
        computes: list[float] = []
        overruns = 0
        policy_errors = 0
        last_cmd: JointCommand | None = None

        clock = time.perf_counter
        t_start = clock()
        prev_start = t_start
        deadline = t_start

        step = 0
        while step < max_steps and not self._stop:
            loop_start = clock()
            if step > 0:
                periods.append(loop_start - prev_start)
            prev_start = loop_start

            state = self.hal.read_state()
            try:
                cmd = self._policy(state)
            except Exception:
                policy_errors += 1
                if self.on_error == "raise":
                    self._safe_stop()
                    raise
                cmd = last_cmd if last_cmd is not None else JointCommand.hold(state.q, kp=0.0, kd=1.0)
            last_cmd = cmd
            self.hal.send_command(cmd)

            work_end = clock()
            compute = work_end - loop_start
            computes.append(compute)

            deadline += self.period
            overran = work_end > deadline
            if overran:
                overruns += 1
                # 밀린 만큼 따라잡지 않고 마감을 현재 시각으로 리셋한다.
                # (안 그러면 한 번 밀렸을 때 폭주하듯 몰아친다)
                deadline = work_end

            if self.logger is not None:
                self.logger.log(
                    state,
                    cmd=cmd,
                    loop_dt=compute,
                    overrun=overran,
                    wall_time=loop_start - self._t_origin,
                )

            step += 1
            if step < max_steps and not self._stop:
                self._wait_until(deadline)
            if duration_sec is not None and clock() - t_start >= duration_sec:
                break

        total = clock() - t_start
        stats = self._build_stats(step, total, periods, computes, overruns, policy_errors)
        self.last_stats = stats
        return stats

    # ------------------------------------------------------------- 내부 구현
    def _wait_until(self, deadline: float) -> None:
        clock = time.perf_counter
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                return
            if remaining > 2e-3:
                # 마감 1.5ms 전까지만 자고, 나머지는 스핀으로 정밀하게 맞춘다.
                time.sleep(remaining - 1.5e-3)
            elif self.spin_wait:
                time.sleep(0)  # GIL 양보만 하고 즉시 복귀
            else:
                time.sleep(remaining)
                return

    def _safe_stop(self) -> None:
        """policy가 터졌을 때 하드웨어를 놓지 않고 부드럽게 세운다."""
        try:
            self.hal.send_command(JointCommand.damping_only(self.hal.n_joints, kd=1.0))
        except Exception:
            pass  # 정지 시도 실패가 원래 예외를 가리면 안 된다

    def _build_stats(self, n_steps, total, periods, computes, overruns, policy_errors) -> LoopStats:
        p = np.asarray(periods, dtype=float) * 1e3
        c = np.asarray(computes, dtype=float) * 1e3
        stats = LoopStats(
            target_hz=self.rate_hz,
            n_steps=n_steps,
            duration_sec=total,
            achieved_hz=(n_steps / total if total > 0 else 0.0),
            overruns=overruns,
            policy_errors=policy_errors,
            periods_ms=p,
        )
        if p.size:
            stats.dt_mean_ms = float(p.mean())
            stats.dt_std_ms = float(p.std())
            stats.dt_min_ms = float(p.min())
            stats.dt_max_ms = float(p.max())
            stats.dt_p50_ms = float(np.percentile(p, 50))
            stats.dt_p95_ms = float(np.percentile(p, 95))
            stats.dt_p99_ms = float(np.percentile(p, 99))
        if c.size:
            stats.compute_mean_ms = float(c.mean())
            stats.compute_p50_ms = float(np.percentile(c, 50))
            stats.compute_p95_ms = float(np.percentile(c, 95))
            stats.compute_max_ms = float(c.max())
        return stats
