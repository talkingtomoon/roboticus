"""델타 보정 노드 — nominal 지령에 tau_ff += Δτ 주입.

그래프 위상:  ... → impedance → delta_corrector → (최종 command)

안전장치 3종 (타협 불가):
  1. 하드 클램프  : |Δτ| ≤ max_frac * torque_limit (기본 30%).
     ('nominal 토크 대비 %'가 아니라 토크 한계 대비다 — 정지 마찰 돌파 순간에는
      nominal이 0이라 nominal 비례 클램프는 정확히 그때 보정을 죽인다)
  2. 페이드인    : 활성화 후 fade_s초에 걸쳐 0→100% 램프
  3. 킬스위치    : disable_correction() 즉시 Δτ=0 (모델/상태 그대로, 출력만 차단)

params(전부 float)는 지름길 ①의 SafetyGuard 화이트리스트에 등록할 수 있다:
    "delta_corrector.gain": ParamSpec(0.0, 1.0, max_abs_step=0.3, max_rel_step=None)
→ LLM 회복 루프가 보정 강도를 낮추는 액션을 낼 수 있다.
"""

from __future__ import annotations

import time

import numpy as np

from robot_core.graph.node import Node
from robot_core.hal.interface import JointCommand


class DeltaCorrectorNode(Node):
    """입력: {"state": JointState, "command": JointCommand}
    출력: {"command": 보정된 JointCommand, "delta_tau": (n,)}

    model이 없거나 킬스위치 상태면 지령을 그대로 통과시킨다 (delta 0).
    """

    def __init__(
        self,
        name: str = "delta_corrector",
        params: dict | None = None,
        enabled: bool = True,
        *,
        model=None,
        torque_limits: np.ndarray | None = None,
    ) -> None:
        defaults = {"gain": 1.0, "max_frac": 0.3, "fade_s": 2.0}
        defaults.update(params or {})
        super().__init__(name, defaults, enabled)
        self.model = model
        self.torque_limits = None if torque_limits is None else \
            np.asarray(torque_limits, dtype=float)

        self._correction_on = False
        self._fade_start_t: float | None = None
        self._infer_us: list[float] = []   # 추론 시간 실측 [µs]
        self.clamp_hits = 0
        self._current_afc = "unknown"      # 현재 하드웨어 AFC 상태 (가드용)
        self.last_refusal: str | None = None

    # -------------------------------------------------------------- 외부 API
    def set_model(self, model, torque_limits=None,
                  current_afc_state: str | None = None) -> None:
        """모델 장착. current_afc_state에 현재 하드웨어의 AFC 상태를 넘길 것.

        모델이 학습한 것은 '수집 당시 남아있던 마찰'이다. AFC(액티브 마찰제거)
        상태가 그때와 다르면 보정이 이중이 되거나 무효가 되므로,
        enable_correction()이 불일치를 확인하고 활성화를 거부한다.
        """
        self.model = model
        if torque_limits is not None:
            self.torque_limits = np.asarray(torque_limits, dtype=float)
        if current_afc_state is not None:
            self._current_afc = str(current_afc_state)
        if hasattr(model, "reset"):
            model.reset()

    def set_current_afc_state(self, state: str) -> None:
        """현재 하드웨어 AFC 상태 갱신 (SDK 조회값을 넘긴다)."""
        self._current_afc = str(state)

    def _afc_mismatch(self) -> str | None:
        """모델↔현재 AFC 불일치 사유 문자열, 없으면 None. unknown은 대조 불가."""
        model_state = str(getattr(self.model, "afc_state", "unknown"))
        if model_state in ("on", "off") and self._current_afc in ("on", "off") \
                and model_state != self._current_afc:
            return (f"model was calibrated with AFC={model_state} but hardware "
                    f"is AFC={self._current_afc}")
        return None

    def enable_correction(self) -> bool:
        """보정 켜기 — 페이드인이 새로 시작된다.

        AFC 상태가 캘리브레이션 당시와 불일치하면 **거부**하고 False를 돌려준다
        (경고는 warnings로도 발행). 그 상태로 보정을 켜면 이중/무효 보정이다 —
        재캘리브레이션하거나 AFC를 수집 당시 상태로 되돌린 뒤 다시 켤 것.
        """
        mismatch = None if self.model is None else self._afc_mismatch()
        if mismatch:
            self.last_refusal = f"correction refused: {mismatch}"
            import warnings
            warnings.warn(self.last_refusal, stacklevel=2)
            self._correction_on = False
            return False

        # 모델은 AFC 상태를 아는데 현재 하드웨어 상태가 미확인이면:
        # 막지는 않되(현장 판단 존중) '검증 안 됨' 흔적을 남기고 경고한다.
        model_state = str(getattr(self.model, "afc_state", "unknown")) \
            if self.model is not None else "unknown"
        if model_state in ("on", "off") and self._current_afc == "unknown":
            self.last_refusal = (f"unverified: model AFC={model_state} but current "
                                 f"hardware AFC state is unknown — verify before trusting")
            import warnings
            warnings.warn(self.last_refusal, stacklevel=2)
        else:
            self.last_refusal = None
        self._correction_on = True
        self._fade_start_t = None  # 첫 update에서 로봇 클록으로 설정
        return True

    def disable_correction(self) -> None:
        """킬스위치: 즉시 Δτ=0. 모델과 노드는 그대로 남는다."""
        self._correction_on = False
        self._fade_start_t = None

    @property
    def correction_enabled(self) -> bool:
        return self._correction_on

    def fade_factor(self, now: float) -> float:
        if not self._correction_on:
            return 0.0
        fade_s = float(self.params["fade_s"])
        if self._fade_start_t is None or fade_s <= 0:
            return 1.0 if fade_s <= 0 else 0.0
        return float(np.clip((now - self._fade_start_t) / fade_s, 0.0, 1.0))

    # ---------------------------------------------------------------- 틱
    def update(self, inputs: dict) -> dict:
        state = inputs["state"]
        cmd: JointCommand = inputs["command"]
        n = len(state.q)

        if not self._correction_on or self.model is None:
            return {"command": cmd, "delta_tau": np.zeros(n)}

        now = float(state.timestamp)
        if self._fade_start_t is None:
            self._fade_start_t = now

        t0 = time.perf_counter()
        delta = self.model.predict(state.q, state.qd, cmd.q_des, cmd.qd_des)
        self._infer_us.append((time.perf_counter() - t0) * 1e6)
        if len(self._infer_us) > 20_000:          # 메모리 상한
            self._infer_us = self._infer_us[-10_000:]

        delta = np.asarray(delta, dtype=float) * float(self.params["gain"]) \
            * self.fade_factor(now)

        # 하드 클램프 (토크 한계 대비 max_frac)
        if self.torque_limits is not None:
            cap = float(self.params["max_frac"]) * self.torque_limits
            clipped = np.clip(delta, -cap, cap)
            if np.any(clipped != delta):
                self.clamp_hits += 1
            delta = clipped

        corrected = JointCommand(
            q_des=cmd.q_des, qd_des=cmd.qd_des,
            tau_ff=cmd.tau_ff + delta,           # 새 배열 — 업스트림 명령 불변
            kp=cmd.kp, kd=cmd.kd,
        )
        return {"command": corrected, "delta_tau": delta}

    # ------------------------------------------------------------- 계측 보고
    def timing_report(self, budget_us_per_joint: float = 50.0) -> str:
        if not self._infer_us:
            return "[CORRECTOR TIMING] (no inference calls yet)"
        v = np.array(self._infer_us)
        n = self.torque_limits.shape[0] if self.torque_limits is not None else 1
        per_joint = v / max(n, 1)
        ok = np.percentile(per_joint, 95) <= budget_us_per_joint
        return (
            f"[CORRECTOR TIMING] {len(v)} calls, model={getattr(self.model, 'model_type', '?')}\n"
            f"  per call : mean {v.mean():.1f} µs | p95 {np.percentile(v, 95):.1f} µs "
            f"| max {v.max():.1f} µs\n"
            f"  per joint: mean {per_joint.mean():.1f} µs | p95 "
            f"{np.percentile(per_joint, 95):.1f} µs  (budget {budget_us_per_joint:.0f} µs"
            f"/joint → {'OK' if ok else 'OVER BUDGET'})\n"
            f"  clamp hits: {self.clamp_hits}"
        )
