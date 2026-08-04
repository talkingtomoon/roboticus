"""캠프 데모용 시나리오 5종 — 전부 결정적 (n_steps 기준, 시드 고정, 벽시계 비의존).

각 시나리오는 ScenarioResult(성공 판정 자동화 + 타임라인)를 돌려준다.
실행: python -m robot_core.integration.scenarios [--scenario S2]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from robot_core.legacy.delta import (
    Collector, ExcitationConfig, PhysicsDeltaModel, SafetyLimits, build_excitation,
)
from robot_core.legacy.full_stack import (
    FullStack, LATERAL_JOINT, StackConfig, TASK,
)
from robot_core.integration.timeline import (
    assert_monotonic, build_timeline, format_timeline,
)
from robot_core.recovery import FailureType

# --- 외란 크기: phact-401 스케일 (감지 임계 tau_detect = 7.2×1.5 = 10.8 Nm) ---
# 임계가 연속 정격 위(10.8)에 있으므로 감지 가능한 충격은 그만큼 커야 한다 —
# 실제 충돌은 짧고 크다 (순간 정격 21.6 안). kd 감쇠를 견디고 8ms+ 초과하려면
# 임계의 ~1.4배 이상 필요.
IMPACT = 15.0          # 측면 충격 [Nm]
IMPACT_DURATION = 0.3  # 충격은 짧다. 길게 밀면 그건 충격이 아니라 지속 저항이고,
                       # 밀린 채 정지한 관절에서 STALL이 (정당하게) 발화해
                       # 회복 루프까지 개입하는 다른 시나리오가 된다 (S3의 영역)
IMPACT_JOINT = LATERAL_JOINT   # j0 베이스 요 — 옆에서 밀리는 축
STALL_FORCE = -1.8     # 지속 저항 [Nm] — 임계(10.8) 미만이라 STALL만이 잡는다
STALL_JOINT = 1        # j1 숄더 — 실제 목표 각도가 있는 축(고착이 의미를 가짐)

# 저강성 시나리오 게인: 처짐 = |STALL_FORCE|/kp = 1.8/7.5 = 0.24 rad
# (STALL 임계 0.08의 3배 — 확실히 잡히되 과하지 않게)
STALL_KP = 7.5
# scripted LLM이 제안할 강성. 7.5 → 18 은 변화율 제한(2.5배) 안이라 그대로 적용된다.
RECOVERY_KP, RECOVERY_KP_RETRY = 18.0, 36.0

# 캘리브레이션 추종 게인 (phact 스케일, ζ≈0.71)
CALIB_KP, CALIB_KD = 12.0, 1.1

# 리허설용 축소 여기 설정. 기본값은 저주파(0.3 Hz) 사인이 들어 있는데
# 생성기의 '최소 1주기' 하한 때문에 관절당 3.3초씩 먹는다 — 6축이면 그것만
# 40초다. 리허설은 파이프라인 검증이 목적이므로 조합을 줄인다.
# **현장(scripts/field_calibration.py)은 기본 설정의 전체 커버리지를 쓴다.**
REHEARSAL_EXCITATION = ExcitationConfig(sweep_freqs=(0.7, 1.4),
                                        sweep_amp_fracs=(0.5,))


def scripted_llm(system: str, user: str) -> str:
    """리허설용 결정적 LLM 대역. STALL엔 kp 상향 제안 (재발 시 더 세게)."""
    kp = RECOVERY_KP_RETRY if "PREVIOUS RECOVERY ATTEMPTS" in user else RECOVERY_KP
    return json.dumps({
        "diagnosis": "stall against sustained resistance; raise stiffness",
        "actions": [{"node": "impedance", "param": "kp", "value": kp}],
        "confidence": 0.9,
    })


@dataclass
class ScenarioResult:
    name: str
    title: str
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    timeline: str = ""
    notes: list[str] = field(default_factory=list)

    def check(self, desc: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((desc, bool(ok), detail))
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def summary(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        lines = [f"[{self.name}] {self.title}  ->  {mark}"]
        for desc, ok, detail in self.checks:
            lines.append(f"  {'v' if ok else 'X'} {desc}"
                         + (f"  ({detail})" if detail else ""))
        for n in self.notes:
            lines.append(f"  - {n}")
        return "\n".join(lines)


def quick_calibrate(stack: FullStack, per_joint_s: float = 4.0,
                    notes: list[str] | None = None) -> PhysicsDeltaModel:
    """축소판 현장 캘리브레이션: 수집(보정기 자동 일시정지) → 물리 피팅 → 적용.

    예산을 '관절당'으로 준다 — 여기 궤적은 순차 모드라 관절 수에 비례해
    길어지므로, 6축에서 총 예산을 고정하면 관절당 데이터가 반토막 난다.

    여기 궤적의 속도 한계는 **프로파일 값(12.56 rad/s)이 아니라 보수적인
    기본값(4.0 → 실사용 2.0)을 그대로 쓴다.** 두 가지 이유:
      1. 마찰 식별에 필요한 것은 저속 구간이지 고속이 아니다. 사인 스윕을
         속도 한계까지 밀면 가속 토크(I*amp*ω²)가 phact의 작은 연속 토크
         예산(5.76 Nm)을 그냥 넘긴다 — 실제로 넘겨서 수집이 중단됐다.
      2. 캘리브레이션은 로봇을 처음 만지는 단계다. 탐침은 얌전할수록 좋다.
    수집기 안전 감시의 qd_abs만 프로파일 값으로 둔다 (하드웨어 상한 가드).
    """
    n = stack.cfg.n_joints
    plan = build_excitation(
        list(range(n)), stack.hal.joint_limits, budget_s=per_joint_s * n,
        n_joints=n, config=REHEARSAL_EXCITATION)
    log_lines: list[str] = []
    # 중단 기준은 연속 예산 절대값 — HAL 한계(순간 21.6)의 비율로 잡으면
    # 저속 탐침에서 뭔가 크게 잘못돼도 18Nm까지 안 끊는다
    collector = Collector(stack.hal, kp=CALIB_KP, kd=CALIB_KD,
                          safety=SafetyLimits(tau_abs=stack.cfg.torque_limit,
                                              qd_abs=stack.cfg.qd_limit),
                          pause_correction=[stack.corrector],
                          on_log=log_lines.append)
    data = collector.collect(plan)
    model = PhysicsDeltaModel(stack.cfg.n_joints).fit(data)
    # Δτ 클램프는 지속 예산(tau_cont) 기준 — HAL 한계(버스트)가 아니다
    stack.corrector.set_model(
        model, torque_limits=np.full(stack.cfg.n_joints, stack.cfg.torque_limit),
        current_afc_state=str(getattr(stack.hal, "afc_state", "unknown")))
    stack.hal.reset()          # 캘리브레이션 흔적을 지우고 로봇 클록 0에서 시작
    stack.logger.clear()
    stack.logger.mark_event(f"calibration done ({data.n_samples} samples, "
                            f"physics model applied)", t=0.0)
    if notes is not None:
        notes.extend(log_lines)
    return model


def _finish(result: ScenarioResult, stack: FullStack, title: str = None) -> ScenarioResult:
    entries = build_timeline(logger=stack.logger, switch_node=stack.chunk_node,
                             guard=stack.guard)
    assert_monotonic(entries)
    result.timeline = format_timeline(entries, title=f"{result.name} TIMELINE")
    return result


# ---------------------------------------------------------------- S1: 평화
def s1_peace() -> ScenarioResult:
    r = ScenarioResult("S1", "평화 — 전 모듈 on, 개입 0 (오발동 검사)")
    stack = FullStack.build(StackConfig())
    quick_calibrate(stack, notes=r.notes)
    stack.corrector.params["fade_s"] = 0.2
    stack.corrector.enable_correction()
    stack.chunk_node.set_active(stack.dictionary.get("direct"), t_now=stack.hal.t)

    stack.step(2000)   # 2.0s: 이동 + 정착

    q = stack.hal.read_state().q
    r.check("추종 성공 (오차 < 0.05)", np.abs(q - stack.cfg.goal).max() < 0.05,
            f"q={np.round(q, 3)}")
    r.check("감지 이벤트 0", not any(e.text.startswith("DETECTED")
                                for e in stack.logger.events()))
    r.check("스위칭 결정 0", len(stack.chunk_node.decisions) == 0)
    r.check("회복 개입 0 (guard 비어있음)", len(stack.guard.audit) == 0)
    r.check("LLM 질의 0", stack.agent.stats.get("submitted", 0) == 0)
    if stack.delta_history:
        peak = float(np.abs(np.array(stack.delta_history)).max())
        cap = 0.3 * stack.cfg.torque_limit
        r.check("보정 토크가 클램프 안", peak <= cap + 1e-9, f"peak={peak:.2f}")
    return _finish(r, stack)


# ---------------------------------------------------------- S2: 충격→우회
def s2_impact_switch() -> ScenarioResult:
    r = ScenarioResult("S2", "충격→우회 — 스위칭만 발동, 회복 루프 침묵 (인터록)")
    stack = FullStack.build(StackConfig())
    stack.chunk_node.set_active(stack.dictionary.get("direct"), t_now=0.0)

    stack.step(400)
    stack.hal.inject_disturbance(IMPACT_JOINT, +IMPACT, duration=IMPACT_DURATION)
    stack.logger.mark_event(
        f"INJECT impact {IMPACT:+.1f} Nm on joint {IMPACT_JOINT}", t=stack.hal.t)
    stack.step(2400)

    switches = [d for d in stack.chunk_node.decisions if d.chosen is not None]
    r.check("스위칭 정확히 1회", len(switches) == 1,
            f"{[d.chosen for d in switches]}")
    if switches:
        # 외란이 j0를 + 로 미니 순응하는 쪽(+j0 스윙)을 골라야 한다
        r.check("순응 우회(detour_left) 선택", switches[0].chosen == "detour_left",
                switches[0].chosen)
    r.check("회복 루프 침묵 (guard 비어있음)", len(stack.guard.audit) == 0)
    r.check("LLM 질의 0", stack.agent.stats.get("submitted", 0) == 0)
    r.check("인터록 'handled by switching' 기록",
            any("handled by switching" in e.text for e in stack.logger.events()))
    q = stack.hal.read_state().q
    r.check("목표 도달", np.abs(q - stack.cfg.goal).max() < 0.06,
            f"q={np.round(q, 3)}")
    return _finish(r, stack)


# ---------------------------------------------------------- S3: 고착→회복
def s3_stall_recovery() -> ScenarioResult:
    r = ScenarioResult("S3", "고착→회복 — STALL은 인터록 예외로 회복 루프 개입")
    stack = FullStack.build(StackConfig(kp=STALL_KP), scripted_llm=scripted_llm)
    stack.chunk_node.set_active(stack.dictionary.get("direct"), t_now=0.0)
    goal_j = stack.cfg.goal[STALL_JOINT]

    stack.step(1600)   # 목표 도달 후 홀드
    stack.hal.inject_disturbance(STALL_JOINT, STALL_FORCE, duration=30.0)
    stack.logger.mark_event(
        f"INJECT sustained {STALL_FORCE:+.1f} Nm on joint {STALL_JOINT}",
        t=stack.hal.t)
    stack.step(600)
    err_before = abs(stack.hal.read_state().q[STALL_JOINT] - goal_j)
    stack.step(1400)
    err_after = abs(stack.hal.read_state().q[STALL_JOINT] - goal_j)

    events = [e.text for e in stack.logger.events()]
    r.check("STALL 감지",
            any(f"DETECTED STALL joint={STALL_JOINT}" in t for t in events))
    r.check("인터록 예외 경로 기록", any("STALL은 인터록 예외" in t for t in events))
    applied = [a for a in stack.guard.audit if a.applied is not None]
    r.check("회복 루프가 파라미터 적용", len(applied) >= 1,
            f"{len(applied)} applies")
    r.check("LLM 경로 사용", any(a.source == "llm" for a in stack.guard.audit))
    r.check("스위칭 미발동", not any(d.chosen for d in stack.chunk_node.decisions))
    r.check("오차 절반 이하로 회복", err_after < err_before / 2,
            f"{err_before:.3f} -> {err_after:.3f} rad")
    return _finish(r, stack)


# ---------------------------------------------------------- S4: 마찰 지옥
def s4_friction_hell() -> ScenarioResult:
    r = ScenarioResult("S4", "마찰 지옥(3배) — 보정 기여도 정량화")
    base = StackConfig()
    hell_tc, hell_b = base.coulomb * 3, base.viscous * 3   # 기본의 3배
    stack = FullStack.build(StackConfig(coulomb=hell_tc, viscous=hell_b))
    model = quick_calibrate(stack, per_joint_s=6.0, notes=r.notes)

    tc = model.params[0].coulomb
    r.check("3배 마찰 복원 (tau_c 오차 < 15%)", abs(tc - hell_tc) / hell_tc < 0.15,
            f"tau_c={tc:.3f} vs {hell_tc:.2f}")

    def track(correct: bool) -> float:
        stack.hal.reset()
        stack.corrector.disable_correction()
        if correct:
            stack.corrector.params["fade_s"] = 0.0
            stack.corrector.enable_correction()
        if hasattr(model, "reset"):
            model.reset()
        stack.chunk_node.set_active(stack.dictionary.get("direct"), t_now=0.0)
        errs = []
        for k in range(1800):
            state = stack.hal.read_state()
            cmd = stack.policy(state)
            stack.hal.send_command(cmd)
            errs.append(np.abs(stack.hal.read_state().q - cmd.q_des))
        return float(np.sqrt(np.mean(np.square(errs))))

    rms_off = track(False)
    rms_on = track(True)
    imp = (1 - rms_on / rms_off) * 100
    r.check("보정 on이 추종 오차 30%+ 개선", imp > 30.0,
            f"off {rms_off:.5f} -> on {rms_on:.5f} rad ({imp:+.1f}%)")
    r.notes.append(f"보정 기여도: RMS {rms_off:.5f} → {rms_on:.5f} rad ({imp:+.1f}%)")
    r.notes.append(stack.corrector.timing_report())
    r.check("추론 예산 준수", "OK" in stack.corrector.timing_report())
    return _finish(r, stack)


# ------------------------------------------------------------ S5: 총력전
def s5_full_battle() -> ScenarioResult:
    r = ScenarioResult("S5", "총력전 — 캘리브레이션→충격→고착 연쇄, 단일 타임라인")
    stack = FullStack.build(StackConfig(kp=STALL_KP), scripted_llm=scripted_llm)

    # --- 0: 캘리브레이션 (보정기가 켜져 있어도 자동 일시정지되는지 = (B)) ---
    stack.corrector.params["fade_s"] = 0.3
    stack.corrector.enable_correction()          # 모델 없이도 '켜짐' 상태
    quick_calibrate(stack, notes=r.notes)
    r.check("(B) 수집 중 보정기 자동 일시정지 로그",
            any("paused for collection" in n for n in r.notes))
    r.check("(B) 수집 후 보정기 복원", stack.corrector.correction_enabled)

    stack.chunk_node.set_active(stack.dictionary.get("direct"), t_now=stack.hal.t)

    # --- 1: 이동 중 측면 충격 → 스위칭 ---
    stack.step(400)
    stack.hal.inject_disturbance(IMPACT_JOINT, +IMPACT, duration=IMPACT_DURATION)
    stack.logger.mark_event(
        f"INJECT impact {IMPACT:+.1f} Nm on joint {IMPACT_JOINT}", t=stack.hal.t)
    stack.step(2400)
    switches = [d for d in stack.chunk_node.decisions if d.chosen is not None]
    r.check("1단계: 충격 → 스위칭 발동", len(switches) == 1,
            f"{[d.chosen for d in switches]}")

    # --- 2: 도달 후 지속 저항 → STALL → 회복 ---
    stack.hal.inject_disturbance(STALL_JOINT, STALL_FORCE, duration=30.0)
    stack.logger.mark_event(
        f"INJECT sustained {STALL_FORCE:+.1f} Nm on joint {STALL_JOINT}",
        t=stack.hal.t)
    stack.step(2000)
    applied = [a for a in stack.guard.audit if a.applied is not None]
    r.check("2단계: STALL → 회복 루프 적용", len(applied) >= 1,
            f"{len(applied)} applies")

    # --- 판정: 타임라인 일관성 ((D)) ---
    entries = build_timeline(logger=stack.logger, switch_node=stack.chunk_node,
                             guard=stack.guard)
    assert_monotonic(entries)
    sources = {e.source for e in entries}
    r.check("(D) 전 소스가 한 타임라인에", {"detect", "interlock", "switch",
                                       "guard", "event"} <= sources,
            f"sources={sorted(sources)}")
    if switches and applied:
        r.check("(D) 순서: 스위칭 → 회복", switches[0].t < applied[0].wall_t,
                f"switch t={switches[0].t:.3f} < guard t={applied[0].wall_t:.3f}")
    # 회복 후 남는 처짐 = |STALL_FORCE| / 상향된 kp = 1.8/18 = 0.10 rad
    err_j = abs(stack.hal.read_state().q[STALL_JOINT] - stack.cfg.goal[STALL_JOINT])
    r.check(f"최종 오차 회복 (joint{STALL_JOINT} < 0.11)", err_j < 0.11,
            f"err={err_j:.3f}")

    r.timeline = format_timeline(entries, title="S5 TIMELINE")
    return r


SCENARIOS = {
    "S1": s1_peace,
    "S2": s2_impact_switch,
    "S3": s3_stall_recovery,
    "S4": s4_friction_hell,
    "S5": s5_full_battle,
}


def run_all(show_timeline: bool = True) -> list[ScenarioResult]:
    results = []
    for name, fn in SCENARIOS.items():
        print(f"\n{'#' * 74}\n# {name} 실행 중...\n{'#' * 74}")
        res = fn()
        results.append(res)
        print(res.summary())
        if show_timeline and (name == "S5" or not res.ok):
            print()
            print(res.timeline)
    print("\n" + "=" * 74)
    n_ok = sum(1 for r in results if r.ok)
    print(f"통합 리허설 결과: {n_ok}/{len(results)} 시나리오 통과"
          + ("  — 전체 시스템 정상" if n_ok == len(results) else "  ** 실패 있음"))
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=list(SCENARIOS), default=None)
    p.add_argument("--no-timeline", action="store_true")
    args = p.parse_args()
    if args.scenario:
        res = SCENARIOS[args.scenario]()
        print(res.summary())
        print(res.timeline)
        raise SystemExit(0 if res.ok else 1)
    results = run_all(show_timeline=not args.no_timeline)
    raise SystemExit(0 if all(r.ok for r in results) else 1)
