"""스위칭 매니저 — 그래프 노드로 등록 가능한 상태기계.

    EXECUTING ──(외란 감지: request_switch)──▶ SCORING(틱 내부) ──▶ BLENDING ──▶ EXECUTING

- request_switch()는 어느 스레드에서 불러도 된다 (모니터 스레드 → 제어 스레드).
  실제 채점·전환은 다음 update() 틱 안에서 일어난다.
- 블렌딩은 항상 "현재 참조 상태"에서 시작하므로 측정 노이즈와 무관하게
  참조 신호가 C1 연속으로 유지된다.
- 쿨다운: 스위칭 직후 params["cooldown_s"] 동안 재스위칭 금지 (채터링 방지).
- 모든 결정(채택/기각 포함)을 감사 로그로 남긴다 — 후보별 점수와 선택 이유.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from robot_core.legacy.chunks.format import MotionChunk
from robot_core.graph.node import Node
from robot_core.legacy.switching.blender import Blender, ChunkPlan
from robot_core.legacy.switching.scorer import ChunkScorer, ScoreReport


@dataclass
class SwitchDecision:
    t: float                     # 로봇 클록
    disturbance: np.ndarray
    report: ScoreReport | None   # None = 쿨다운 등으로 채점 자체를 안 함
    chosen: str | None           # None = 스위칭 안 함
    reason: str
    blend_s: float = 0.0

    def line(self) -> str:
        d = np.array2string(self.disturbance, precision=2, suppress_small=True)
        out = f"[t={self.t:8.3f}] d={d} -> "
        if self.chosen is None:
            return out + f"NO SWITCH ({self.reason})"
        return out + f"SWITCH to {self.chosen!r} (blend {self.blend_s * 1e3:.0f} ms, {self.reason})"


class ChunkSwitchNode(Node):
    """현재 계획(청크 또는 전이+청크)을 따라 (q_des, qd_des)를 내보내는 노드.

    입력:  {"state": JointState}
    출력:  {"q_des": (n,), "qd_des": (n,)}  → ImpedanceNode가 지령으로 변환

    사용법:
        node = ChunkSwitchNode(scorer=scorer, goal=goal)
        node.set_active(chunk, t_now)         # 초기 궤적 시작
        node.request_switch(disturbance_vec)  # 모니터 스레드에서 (논블로킹)
    """

    def __init__(
        self,
        name: str = "chunk_switch",
        params: dict | None = None,
        enabled: bool = True,
        *,
        scorer: ChunkScorer | None = None,
        blender: Blender | None = None,
        goal: np.ndarray | None = None,
    ) -> None:
        # time_scale: 재생 속도 (규칙/LLM 폴백의 "동작 속도 하향" 손잡이).
        # 위상 적분 방식이라 값이 바뀌어도 위치는 절대 점프하지 않고,
        # 내부 저역통과(_SCALE_LP_TAU)가 속도 점프도 없앤다.
        defaults = {"cooldown_s": 0.4, "time_scale": 1.0}
        defaults.update(params or {})
        super().__init__(name, defaults, enabled)
        self.scorer = scorer
        self.blender = blender or Blender()
        self.goal = None if goal is None else np.asarray(goal, dtype=float)

        self._plan = None            # ChunkPlan | TransitionPlan
        self._phase = 0.0            # 플랜 시간 [s] — ∫ scale dt (실시간 아님)
        self._last_now = None        # 위상 적분용 직전 로봇 클록
        self._scale = 1.0            # 저역통과된 실효 스케일
        self._last_switch_t = -1e9
        self._pending: np.ndarray | None = None
        self._lock = threading.Lock()
        self.decisions: list[SwitchDecision] = []

    _SCALE_LP_TAU = 0.3      # [s] scale 변경 저역통과 (속도 점프 방지)
    _SCALE_MIN, _SCALE_MAX = 0.2, 1.5

    # -------------------------------------------------------------- 외부 API
    def set_active(self, chunk: MotionChunk, t_now: float = 0.0,
                   goal: np.ndarray | None = None) -> None:
        """초기(또는 강제) 궤적 설정. goal 미지정이면 청크 종점을 목표로 삼는다."""
        self._plan = ChunkPlan(chunk)
        self._phase = 0.0
        self._last_now = float(t_now)
        if goal is not None:
            self.goal = np.asarray(goal, dtype=float)
        elif self.goal is None:
            self.goal = chunk.q_end.copy()

    def request_switch(self, disturbance: np.ndarray) -> None:
        """스위칭 요청 (스레드 안전, 논블로킹). 다음 틱에서 처리된다.

        연속 요청이 오면 마지막 것만 남는다 — 어차피 최신 외란 추정이 정답이다.
        """
        with self._lock:
            self._pending = np.asarray(disturbance, dtype=float).copy()

    @property
    def phase(self) -> str:
        if self._plan is None:
            return "IDLE"
        if self._phase >= self._plan.duration:
            return "DONE"
        if self._phase < self._plan.blend_s:
            return "BLENDING"
        return "EXECUTING"

    @property
    def effective_time_scale(self) -> float:
        """저역통과 후 실제 적용 중인 재생 속도."""
        return self._scale

    def dump_decisions(self) -> str:
        if not self.decisions:
            return "[SWITCH AUDIT] (no decisions)"
        lines = ["[SWITCH AUDIT]"]
        for d in self.decisions:
            lines.append("  " + d.line())
            if d.report is not None:
                for row in d.report.table(top=5).splitlines():
                    lines.append("    " + row)
        return "\n".join(lines)

    # ---------------------------------------------------------------- 틱
    def update(self, inputs: dict) -> dict:
        """위상 적분 재생: phase += dt × scale.

        시간 재매핑을 t0 기준으로 하면 scale 변경 순간 위치가 점프하지만,
        위상 누적은 정의상 연속이다. 출력 속도는 체인룰로 plan 속도 × scale.
        전이 스플라인과 본 궤적 모두 같은 위상으로 돈다 — 시간축이 하나라
        scale이 언제 바뀌어도 경계 C1 논증이 한 규칙으로 끝난다.
        """
        state = inputs["state"]
        now = float(state.timestamp)

        if self._plan is None:
            # 활성 궤적 없음: 현재 자세 홀드 (안전 기본값)
            self._last_now = now
            return {"q_des": state.q.copy(), "qd_des": np.zeros_like(state.q)}

        # scale 저역통과 + 위상 적분
        target = float(np.clip(float(self.params["time_scale"]),
                               self._SCALE_MIN, self._SCALE_MAX))
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now
        if dt > 0:
            alpha = min(1.0, dt / self._SCALE_LP_TAU)
            self._scale += (target - self._scale) * alpha
        self._phase += dt * self._scale

        request = self._take_request()
        if request is not None:
            self._handle_request(request, now, self._phase)
            # 전환됐으면 _handle_request가 _phase를 0으로 리셋했다

        q_des, qd_des, _ = self._plan.sample(self._phase)
        # 체인룰: dq/dt = dq/dphase × dphase/dt = qd_plan × scale
        return {"q_des": q_des, "qd_des": qd_des * self._scale}

    # ------------------------------------------------------------- 내부 구현
    def _take_request(self) -> np.ndarray | None:
        with self._lock:
            req, self._pending = self._pending, None
            return req

    def _handle_request(self, disturbance: np.ndarray, now: float, phase: float) -> None:
        cooldown = float(self.params["cooldown_s"])
        if now - self._last_switch_t < cooldown:
            self.decisions.append(SwitchDecision(
                t=now, disturbance=disturbance, report=None, chosen=None,
                reason=f"cooldown ({now - self._last_switch_t:.2f}s < {cooldown:.2f}s)"))
            return
        if self.scorer is None:
            self.decisions.append(SwitchDecision(
                t=now, disturbance=disturbance, report=None, chosen=None,
                reason="no scorer configured"))
            return

        # 블렌딩 출발점은 '현재 참조 상태' — 참조 신호의 C1 연속성 보장.
        # 경계조건은 플랜 공간 속도(스케일 곱하기 전)로 준다: 새 플랜도 같은
        # scale로 재생되므로 출력 속도 = qd_plan × scale 이 경계에서 일치한다.
        q_ref, qd_ref, qdd_ref = self._plan.sample(phase)
        report = self.scorer.score(q_ref, qd_ref, disturbance, self.goal)

        if report.best is None:
            self.decisions.append(SwitchDecision(
                t=now, disturbance=disturbance, report=report, chosen=None,
                reason="all candidates vetoed (predicted torque limit) — keeping current"))
            return

        chunk = self.scorer.chunks[report.best_index]
        plan = self.blender.blend(q_ref, qd_ref, chunk, qdd=qdd_ref)
        self._plan = plan
        self._phase = 0.0
        self._last_switch_t = now
        self.decisions.append(SwitchDecision(
            t=now, disturbance=disturbance, report=report, chosen=chunk.name,
            reason=f"best total={report.best.total:.3f}, scored in {report.elapsed_ms:.2f} ms",
            blend_s=plan.blend_s))
