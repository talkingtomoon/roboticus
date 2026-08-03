"""세 지름길을 한 그래프에 올리는 표준 조립.

그래프 위상 (물리적 근거):

    target ──▶ chunk_switch ──▶ impedance ──▶ delta_corrector ──▶ (HAL)

  1. target        : 기본 목표 생성. chunk_switch가 꺼져도 시스템이 동작하는
                     폴백 (아래로도 간선을 이어 둔다).
  2. chunk_switch  : 참조(q_des, qd_des)를 덮어쓴다. 위상순서상 target보다
                     뒤이므로 merge에서 이긴다. 반드시 임피던스 '앞' —
                     참조가 바뀌면 그에 대한 nominal 토크도 다시 계산돼야 한다.
  3. impedance     : 참조 → nominal 지령 (kp/kd/tau_ff=0).
  4. delta_corrector: nominal이 확정된 뒤에야 마찰 보정 tau_ff를 얹을 수 있다.
                     맨 마지막 = HAL 직전. 이 위치라야 클램프가 '최종 지령'에
                     대한 마지막 방어선이 된다.

(A) 이중 반응 인터록 — 그래프 매니저 수준 메커니즘:
  모듈들은 서로를 모른다 (직접 import 없음). FullStack.monitor_poll()이
  mgr.node("chunk_switch").phase 를 읽어 이벤트를 라우팅한다:

  - TORQUE_SPIKE → ms 경로(스위칭)가 먼저 가져간다. 다음 폴에서 스위칭
    결정을 확인해, 스위칭이 채택됐으면 회복 루프엔 넘기지 않는다("handled").
    기각(전원 veto/쿨다운)됐으면 그때 회복 루프로 넘긴다.
    단, 처짐이 줄어드는 중이면 '외란 해제 잔향'이므로 아예 반응하지 않는다.
  - STALL → 항상 회복 루프로 (스위칭으로 해결 안 되는 실패).
  - 그 외(OSCILLATION 등) → 회복 루프로. 단 BLENDING 중이면 보류(defer)했다가
    전이가 끝나면 제출한다.
  모든 라우팅 결정은 logger.mark_event로 남아 통합 타임라인에 찍힌다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robot_core.hal.mock import MockRobotHAL
from robot_core.hal.profiles import PHACT_401
from robot_core.logging.ring_logger import RingLogger
from robot_core.graph import (
    ImpedanceNode, NodeGraphManager, TargetNode,
)
from robot_core.chunks import ChunkDictionary, generator as G
from robot_core.switching import (
    ChunkScorer, ChunkSwitchNode, DreamModel, ScorerConfig, estimate_disturbance,
)
from robot_core.recovery import (
    DetectorConfig, FailureDetector, FailureEvent, FailureType, LLMConfig,
    LLMRecoveryAgent, ParamSpec, RuleBasedRecovery, SafetyGuard,
)
from robot_core.delta import DeltaCorrectorNode

# YAML 왕복용 노드 타입 레지스트리 (scorer/model 같은 런타임 객체는 YAML에
# 들어가지 않는다 — 로드 후 set_active/set_model로 다시 붙인다)
NODE_TYPES = {
    "TargetNode": TargetNode,
    "ImpedanceNode": ImpedanceNode,
    "ChunkSwitchNode": ChunkSwitchNode,
    "DeltaCorrectorNode": DeltaCorrectorNode,
}

# ---------------------------------------------------- phact-401 6축 포즈/딕셔너리
# 미션 미공개 상태이므로 '내용'이 아니라 '형식'이다. 미션이 공개되면 아래 세
# 상수(HOME / TASK / LATERAL_JOINT)만 실제 값으로 갈아끼우면 딕셔너리 전체가
# 따라 바뀐다 — README "딕셔너리 채우는 절차" 참고.
#
# 축 배치 (6축 관절 팔의 통상 구성):
#   j0 base yaw | j1 shoulder pitch | j2 elbow pitch
#   j3 wrist roll | j4 wrist pitch | j5 wrist yaw
HOME = np.zeros(6)
# 앞으로 뻗어 작업점에 접근하는 자세. 접근은 j1/j2/j4(피치 체인)가 만들고
# j0(베이스 요)는 0을 유지한다 — 즉 j0가 '옆으로 비켜설 여유 축'이다.
TASK = np.array([0.0, -0.55, 0.90, 0.0, 0.45, 0.0])

LATERAL_JOINT = 0      # 우회(회피)가 일어나는 축. 측면 외란도 이 축으로 들어온다
DETOUR_OFFSET = 0.35   # [rad] 우회 범프 크기

# 게인은 액추에이터 토크 스케일에 비례해야 한다. phact-401 연속 한계 5.76 Nm
# 기준: kp=18 (관성 0.05에서 ωn≈19 rad/s), kd=1.35 (ζ≈0.71).
KP, KD = 18.0, 1.35


def default_dictionary(start=None, goal=None, *,
                       lateral_joint: int = LATERAL_JOINT,
                       offset: float = DETOUR_OFFSET) -> ChunkDictionary:
    """phact-401 6축 기본 딕셔너리 (관절 수는 start/goal 길이를 따른다).

    직진 1 + 좌우 우회 2 + 저속 변주 1 + 후퇴 1 = 5개.
    우회는 lateral_joint(베이스 요)를 ± 로 스윙해 장애물을 옆으로 돌아간다.
    """
    # 관절 수는 goal이 정한다. start를 생략하면 같은 길이의 HOME(=원점).
    goal = TASK.copy() if goal is None else np.asarray(goal, dtype=float)
    n = len(goal)
    start = np.zeros(n) if start is None else np.asarray(start, dtype=float)
    if len(start) != n:
        raise ValueError(f"start({len(start)})와 goal({n})의 관절 수가 다르다")
    off = np.zeros(n)
    off[lateral_joint] = offset

    direct = G.min_jerk("direct", start, goal, 1.2, tags=["approach"])
    return ChunkDictionary([
        direct,
        G.with_detour(direct, t_via=0.6, offset=+off, name="detour_left"),
        G.with_detour(direct, t_via=0.6, offset=-off, name="detour_right"),
        G.time_scaled(direct, 1.6, name="direct_slow"),
        G.retreat("retreat", goal * 0.5, start, 1.0),
    ])


@dataclass
class StackConfig:
    """기본값 = phact-401 6축. 토크 한계는 용도별 3단 (profiles.py 원칙 참고):

    - torque_limit (tau_cont, 5.76)  : 지속 신호 예산 — 보정기 Δτ 클램프 기준,
                                       1초 이동평균 과부하 감시, 수집 중단 기준
    - torque_limit_clamp (21.6)      : HAL 포화(최종 출력 클램프) — 순간 상한
    - tau_detect (10.8)              : TORQUE_SPIKE 감지 임계
    - torque_limit_peak (21.6)       : dream veto 전용
    """

    n_joints: int = PHACT_401.n_joints
    torque_limit: float = PHACT_401.tau_cont
    torque_limit_clamp: float = PHACT_401.tau_clamp
    tau_detect: float = PHACT_401.tau_detect
    # 순간 정격 기반 한계 — 스위칭 dream veto 전용 (연속 지령에 쓰지 말 것).
    # None이면 HAL 클램프 기준으로 veto한다.
    torque_limit_peak: float | None = PHACT_401.tau_veto
    qd_limit: float = PHACT_401.qd_limit
    # 목의 마찰/백래시는 phact 토크 스케일에 맞춘 플레이스홀더다 (AFC off 기준).
    # 실물의 잔여 마찰은 캘리브레이션이 측정한다 — 여기 값은 리허설용일 뿐.
    coulomb: float = 0.12
    viscous: float = 0.045
    backlash: float = 0.02
    kp: float = KP
    kd: float = KD
    goal: np.ndarray = field(default_factory=lambda: TASK.copy())
    monitor_every_steps: int = 20          # 제어 20스텝(=20ms)마다 모니터 1회
    torque_threshold_frac: float = 0.45    # 감지 임계 = 한계 * frac
    stall_err_threshold: float = 0.08
    llm_cooldown_s: float = 1.0
    # 결정적 리허설: 회복 이벤트를 모니터 폴 안에서 동기 처리.
    # 실물 운용에서는 False로 두고 agent.start() 워커 스레드를 쓴다.
    sync_recovery: bool = True

    def __post_init__(self):
        # goal 길이를 n_joints에 맞춘다 (관절 수를 줄여 빠르게 돌릴 때 대비).
        g = np.asarray(self.goal, dtype=float)
        if len(g) != self.n_joints:
            g = (g[:self.n_joints] if len(g) > self.n_joints
                 else np.concatenate([g, np.zeros(self.n_joints - len(g))]))
        self.goal = g

    @classmethod
    def from_profile(cls, profile, n_joints: int | None = None, **overrides) -> "StackConfig":
        """하드웨어 프로파일 → 스택 설정. n_joints로 관절 수를 줄일 수 있다
        (goal은 __post_init__이 자동으로 맞춰 자른다)."""
        return cls(
            n_joints=n_joints if n_joints is not None else profile.n_joints,
            torque_limit=profile.tau_cont,            # 지속 예산
            torque_limit_clamp=profile.tau_clamp,     # HAL 포화 (순간 상한)
            tau_detect=profile.tau_detect,            # 충돌 감지 임계
            torque_limit_peak=profile.tau_veto,       # dream veto 전용
            qd_limit=profile.qd_limit,
            **overrides,
        )


class FullStack:
    """조립된 전체 스택 + 결정적 실행 루프 + 이벤트 라우팅(인터록)."""

    def __init__(self, cfg: StackConfig, hal, mgr, logger, detector, guard,
                 agent, dictionary, rules=None) -> None:
        self.cfg = cfg
        self.hal = hal
        self.mgr = mgr
        self.logger = logger
        self.detector = detector
        self.guard = guard
        self.agent = agent
        self.dictionary = dictionary
        self.rules = rules   # OVERLOAD_CLEARED 직행 경로용 (LLM/쿨다운 우회)

        self._deferred: list[FailureEvent] = []
        self._pending_switch: FailureEvent | None = None
        self.delta_history: list[np.ndarray] = []
        self._step_count = 0

    # ------------------------------------------------------------------ 조립
    @classmethod
    def build(cls, cfg: StackConfig | None = None, *, client=None,
              scripted_llm=None) -> "FullStack":
        cfg = cfg or StackConfig()
        # HAL 포화 = 순간 상한(tau_clamp). 버스트(충격 반응, 전이)는 이 대역을
        # 쓰는 것이 설계 의도다 — 연속 예산은 별도 이동평균 감시가 지킨다.
        hal = MockRobotHAL(
            n_joints=cfg.n_joints, dt=1e-3, torque_limit=cfg.torque_limit_clamp,
            coulomb_friction=cfg.coulomb, viscous_friction=cfg.viscous,
            backlash_width=cfg.backlash, enable_backlash=cfg.backlash > 0)
        # 로거 임계 표시는 감지 임계와 맞추고, tau_cmd 클램프는 HAL 한계
        logger = RingLogger.for_hal(
            hal, window_sec=8.0,
            threshold_frac=cfg.tau_detect / cfg.torque_limit_clamp)

        dictionary = default_dictionary(goal=cfg.goal)
        dream = DreamModel.from_mock_hal(hal, kp=cfg.kp, kd=cfg.kd)
        if cfg.torque_limit_peak is not None:
            # dream veto는 순간 정격 기준 — 전이 스플라인의 짧은 과도 토크가
            # 연속 정격을 넘는 건 허용된다 (phact-401: 연속 5.76 / 순간 21.6 Nm)
            dream.torque_limit = np.full(cfg.n_joints, float(cfg.torque_limit_peak))
        scorer = ChunkScorer(dictionary.all(), dream=dream,
                             config=ScorerConfig(entry_dir_window_s=0.6))

        mgr = NodeGraphManager()
        mgr.add_node(TargetNode(params={"depth": float(cfg.goal[0]), "retreat": 0.0}))
        mgr.add_node(ChunkSwitchNode(params={"cooldown_s": 0.4}, scorer=scorer,
                                     goal=cfg.goal))
        mgr.add_node(ImpedanceNode(params={"kp": cfg.kp, "kd": cfg.kd}))
        # 보정기 Δτ 클램프는 **연속 예산** 기준 — Δτ는 지속 신호라 버스트
        # 대역(tau_clamp)을 쓰면 열 예산을 무기한 초과 주입할 수 있다
        mgr.add_node(DeltaCorrectorNode(
            torque_limits=np.full(cfg.n_joints, cfg.torque_limit)))
        # 위상: target → chunk_switch → impedance → delta_corrector
        # (target→impedance 간선은 chunk_switch가 꺼졌을 때의 폴백 경로)
        mgr.connect("target", "chunk_switch")
        mgr.connect("target", "impedance")
        mgr.connect("chunk_switch", "impedance")
        mgr.connect("impedance", "delta_corrector")

        # 회복 루프: SafetyGuard 화이트리스트에 세 모듈의 손잡이를 전부 등록
        guard = SafetyGuard(mgr, {
            "impedance.kp": ParamSpec(1.0, 120.0, max_rel_step=2.5),
            "impedance.kd": ParamSpec(0.1, 8.0, max_rel_step=2.5),
            "target.depth": ParamSpec(0.0, 1.0, max_rel_step=None, max_abs_step=0.2),
            "target.retreat": ParamSpec(0.0, 0.3, max_rel_step=None, max_abs_step=0.1),
            "delta_corrector.gain": ParamSpec(0.0, 1.0, max_rel_step=None,
                                              max_abs_step=0.5),
            # 동작 속도 하향/복원 (CONTINUOUS_OVERLOAD 대응). 노드가 내부
            # 저역통과로 부드럽게 적용하므로 step 제한은 널널하게.
            "chunk_switch.time_scale": ParamSpec(0.3, 1.0, max_rel_step=None,
                                                 max_abs_step=0.7),
        }, time_fn=lambda: hal.t)  # 감사 로그를 로봇 클록으로 → 단일 타임라인

        rules = RuleBasedRecovery(mgr)
        agent = LLMRecoveryAgent(
            guard, rules, logger=logger,
            config=LLMConfig(timeout_s=2.0, cooldown_s=cfg.llm_cooldown_s),
            client=scripted_llm or client,
            time_fn=lambda: hal.t)  # 쿨다운도 로봇 클록 → 결정적 시나리오

        # min_duration 8ms: 충격 과도신호는 kd 댐핑이 ms 단위로 깎아 짧다.
        # 감지 임계는 tau_detect(연속×1.5) — 자체 유발 과도 토크(전이 스플라인)
        # 위에 있어야 자기 동작을 충돌로 오검출하지 않는다.
        # cont_budget: 지령 토크 1초 이동평균 예산 → CONTINUOUS_OVERLOAD.
        detector = FailureDetector(cfg.n_joints, DetectorConfig(
            torque_threshold=np.full(cfg.n_joints, cfg.tau_detect),
            torque_min_duration_s=0.008,
            stall_err_threshold=cfg.stall_err_threshold,
            stall_min_duration_s=0.25,
            cont_budget=np.full(cfg.n_joints, cfg.torque_limit),
            refractory_s=0.8))

        return cls(cfg, hal, mgr, logger, detector, guard, agent, dictionary,
                   rules=rules)

    # ------------------------------------------------------------- 노드 접근
    @property
    def chunk_node(self) -> ChunkSwitchNode:
        return self.mgr.node("chunk_switch")

    @property
    def corrector(self) -> DeltaCorrectorNode:
        return self.mgr.node("delta_corrector")

    # ------------------------------------------------------------- 제어 루프
    def policy(self, state):
        out = self.mgr.step({"state": state})
        if out.get("delta_corrector"):
            self.delta_history.append(out["delta_corrector"]["delta_tau"])
            return out["delta_corrector"]["command"]
        return out["impedance"]["command"]  # 보정기 노드가 꺼진 경우 폴백

    def step(self, n_steps: int, monitor: bool = True) -> None:
        """결정적 실행: n_steps 제어 틱 + monitor_every_steps마다 모니터 폴.

        (실물에서는 ControlLoopRunner + 모니터 스레드가 이 역할을 한다.
        리허설은 벽시계와 무관하게 같은 결과가 나와야 하므로 단일 루프로 돈다.)
        """
        for _ in range(n_steps):
            state = self.hal.read_state()
            cmd = self.policy(state)
            self.hal.send_command(cmd)
            self.logger.log(state, cmd=cmd, loop_dt=1e-4, wall_time=state.timestamp)
            self._step_count += 1
            if monitor and self._step_count % self.cfg.monitor_every_steps == 0:
                self.monitor_poll()

    # -------------------------------------------------- (A) 모니터 + 인터록
    def monitor_poll(self) -> list[FailureEvent]:
        # 지난 폴의 스위칭 요청 결과를 먼저 정산
        self._settle_pending_switch()

        events = self.detector.check(self.logger.to_arrays(window_sec=1.5))
        for ev in events:
            self.logger.mark_event(f"DETECTED {ev.describe()}", t=ev.t)
            self._route(ev)

        # BLENDING이 끝났으면 보류된 이벤트를 회복 루프로 방출
        if self._deferred and self.chunk_node.phase != "BLENDING":
            for ev in self._deferred:
                self.logger.mark_event(
                    f"interlock: released deferred {ev.type.value} j{ev.joint_idx} "
                    f"to recovery")
                self._submit_recovery(ev)
            self._deferred.clear()
        return events

    def _route(self, ev: FailureEvent) -> None:
        phase = self.chunk_node.phase

        if ev.type == FailureType.OVERLOAD_CLEARED:
            # 실패가 아니라 상태 전이 — LLM/쿨다운을 우회하고 규칙에 직행.
            # (낮췄던 동작 속도의 복원. 내려가는 길만 있으면 안 된다.)
            if self.rules is not None:
                actions = self.rules.propose(ev)
                entries = self.guard.apply(actions, source="rules") if actions else []
                applied = [e for e in entries if e.applied is not None]
                self.logger.mark_event(
                    f"overload cleared j{ev.joint_idx} — restore via rules "
                    f"({len(applied)} applied)")
            return

        if ev.type == FailureType.TORQUE_SPIKE and self.chunk_node.scorer is not None:
            # 외란 '해제'도 반대 부호의 충격처럼 보인다 (힘이 사라지는 순간
            # 구동 토크만 남아 스파이크). 이미 사라진 외란에 스위칭으로 반응하면
            # 안 된다 — 처짐이 줄어드는 중이면(참조로 복귀 중) 잔향으로 판정.
            trend = self._deflection_trend(ev.joint_idx)
            if trend < -0.01:
                self.logger.mark_event(
                    f"interlock: TORQUE_SPIKE j{ev.joint_idx} ignored — "
                    f"disturbance already released (deflection shrinking "
                    f"{trend:+.3f} rad)")
                return

            # ms 경로 우선: 스위칭에 넘기고, 채택 여부는 다음 폴에서 정산
            d = estimate_disturbance(self.logger.to_arrays(window_sec=1.5),
                                     window_s=0.05, baseline_s=0.4,
                                     stiffness=self.cfg.kp)
            self.chunk_node.request_switch(d)
            self._pending_switch = ev
            self.logger.mark_event(
                f"interlock: TORQUE_SPIKE j{ev.joint_idx} routed to switching "
                f"(d={np.array2string(d, precision=2)})")
            return

        if ev.type == FailureType.STALL:
            # 예외 경로: 스위칭으로 해결 안 되는 실패 — BLENDING 중이라도 통과
            self.logger.mark_event(
                f"interlock: STALL j{ev.joint_idx} passed to recovery "
                f"(phase={phase}, STALL은 인터록 예외)")
            self._submit_recovery(ev)
            return

        if phase == "BLENDING":
            self._deferred.append(ev)
            self.logger.mark_event(
                f"interlock: {ev.type.value} j{ev.joint_idx} DEFERRED "
                f"(switching busy: BLENDING)")
            return

        self._submit_recovery(ev)

    def _deflection_trend(self, joint: int) -> float:
        """관절의 |q - q_des| 최근(50ms) 평균 − 과거(150~300ms 전) 평균.

        양수 = 처짐 증가 중(새 외란이 밀고 있음), 음수 = 참조로 복귀 중
        (외란이 사라진 뒤의 잔향). 데이터 부족이면 0 (= 반응 허용).
        """
        a = self.logger.to_arrays(window_sec=0.4)
        t, q, q_des = a["t"], a["q"], a["q_des"]
        if len(t) < 50 or not np.all(np.isfinite(q_des[:, joint])):
            return 0.0
        defl = np.abs(q[:, joint] - q_des[:, joint])
        t_end = t[-1]
        recent = t >= t_end - 0.05
        old = (t >= t_end - 0.30) & (t <= t_end - 0.15)
        if not old.any():
            return 0.0
        return float(defl[recent].mean() - defl[old].mean())

    def _settle_pending_switch(self) -> None:
        """직전 폴에서 스위칭에 넘긴 TORQUE_SPIKE의 처리 결과 정산."""
        if self._pending_switch is None:
            return
        ev, self._pending_switch = self._pending_switch, None
        decisions = self.chunk_node.decisions
        if decisions and decisions[-1].chosen is not None:
            self.logger.mark_event(
                f"interlock: TORQUE_SPIKE j{ev.joint_idx} handled by switching "
                f"({decisions[-1].chosen!r}) — recovery not engaged")
        else:
            reason = decisions[-1].reason if decisions else "no decision recorded"
            self.logger.mark_event(
                f"interlock: switching declined ({reason}) — escalating "
                f"TORQUE_SPIKE j{ev.joint_idx} to recovery")
            self._submit_recovery(ev)

    def _submit_recovery(self, ev: FailureEvent) -> None:
        accepted = self.agent.submit(ev)
        if not accepted:
            self.logger.mark_event(
                f"recovery submit dropped (cooldown/queue) {ev.type.value} "
                f"j{ev.joint_idx}")
            return
        if self.cfg.sync_recovery:
            # 결정적 리허설 모드: 이 자리에서 즉시 처리 (실물은 워커 스레드)
            self.agent.process_pending()

    # ------------------------------------------------------------------ YAML
    def save_graph_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        self.mgr.save_yaml(path)
        return path

    @staticmethod
    def load_graph_yaml(path: str | Path) -> NodeGraphManager:
        """YAML → 그래프. scorer/model 등 런타임 객체는 호출측이 다시 붙인다."""
        return NodeGraphManager.from_yaml(path, NODE_TYPES)
