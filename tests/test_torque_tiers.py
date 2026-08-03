"""토크 한계 3단 분리 + CONTINUOUS_OVERLOAD + time_scale + AFC 선언."""

import numpy as np
import pytest

from robot_core import JointCommand, JointState, MockRobotHAL, RingLogger
from robot_core.graph import ImpedanceNode, NodeGraphManager
from robot_core.hal import PHACT_401
from robot_core.integration import FullStack, StackConfig
from robot_core.integration.full_stack import KD, KP, TASK, default_dictionary
from robot_core.recovery import (
    DetectorConfig, FailureDetector, FailureEvent, FailureType,
)
from robot_core.switching import (
    BlendConfig, Blender, ChunkScorer, ChunkSwitchNode, DreamModel, ScorerConfig,
)
from scripts.field_calibration import resolve_afc_state

N = PHACT_401.n_joints


# ---------------------------------------------------------------- 프로파일 3단
def test_profile_three_tier_values():
    p = PHACT_401
    assert p.tau_clamp == pytest.approx(27.0 * 0.8)    # 21.6 — 최종 출력 클램프
    assert p.tau_detect == pytest.approx(7.2 * 1.5)    # 10.8 — 감지 임계
    assert p.tau_veto == pytest.approx(27.0 * 0.8)     # 21.6 — dream veto
    assert p.tau_cont == pytest.approx(7.2 * 0.8)      # 5.76 — 지속 예산
    # 하위 호환 별칭
    assert p.tau_limit == p.tau_cont
    assert p.tau_limit_peak == p.tau_veto


# ------------------------------------------- 전이 오검출 대조 (요구사항 2)
@pytest.fixture(scope="module")
def blend_transient_arrays():
    """무거운 관성(0.5) + 3배 마찰 목에서 공격적 전이를 유발해 기록한 로그.

    외란 주입 없음 — 이 로그의 모든 토크는 자체 유발이다.
    실측 피크 ~7.6 Nm: 연속 예산(5.76) 위, 감지 임계(10.8) 아래.
    """
    hal = MockRobotHAL(n_joints=N, dt=1e-3, torque_limit=PHACT_401.tau_clamp,
                       inertia=0.5, coulomb_friction=0.36, viscous_friction=0.135)
    d = default_dictionary()
    dream = DreamModel.from_mock_hal(hal, kp=KP, kd=KD)
    dream.torque_limit = np.full(N, PHACT_401.tau_veto)
    node = ChunkSwitchNode(
        scorer=ChunkScorer(d.all(), dream=dream,
                           config=ScorerConfig(entry_dir_window_s=0.6)),
        blender=Blender(BlendConfig(k_pos=0.8, k_vel=0.15)), goal=TASK)
    node.set_active(d.get("direct"), t_now=0.0)
    mgr = NodeGraphManager()
    mgr.add_node(node)
    mgr.add_node(ImpedanceNode(params={"kp": KP, "kd": KD}))
    mgr.connect("chunk_switch", "impedance")

    logger = RingLogger.for_hal(hal, window_sec=5.0)
    for k in range(2600):
        s = hal.read_state()
        cmd = mgr.step({"state": s})["impedance"]["command"]
        hal.send_command(cmd)
        logger.log(s, cmd=cmd, loop_dt=1e-4, wall_time=s.timestamp)
        if k == 600:
            dd = np.zeros(N); dd[0] = 5.0
            node.request_switch(dd)
    assert any(x.chosen for x in node.decisions), "전이가 실제로 일어나야 한다"
    return logger.to_arrays()


def _poll_detector(arrays, threshold):
    """실제 모니터처럼 20ms 폴링을 재현 (감지기는 상태를 가지므로)."""
    det = FailureDetector(N, DetectorConfig(
        torque_threshold=threshold, torque_min_duration_s=0.008, refractory_s=0.8))
    t = arrays["t"]
    events = []
    for t_end in np.arange(t[0] + 0.5, t[-1], 0.02):
        m = t <= t_end
        sub = {k: v[m] for k, v in arrays.items()}
        events += det.check(sub)
    return [e for e in events if e.type == FailureType.TORQUE_SPIKE]


def test_blend_transient_fires_at_continuous_threshold(blend_transient_arrays):
    """대조 A: 연속 한계(5.76)를 감지 임계로 쓰면 자기 전이를 충돌로 오검출한다."""
    spikes = _poll_detector(blend_transient_arrays, PHACT_401.tau_cont)
    assert len(spikes) >= 1, "연속 기준이면 전이 과도 토크(~7.6Nm)에 발화해야 한다"


def test_blend_transient_silent_at_detect_threshold(blend_transient_arrays):
    """대조 B: tau_detect(10.8 = 연속×1.5)면 같은 로그에서 침묵한다 — 3단 분리의 근거."""
    spikes = _poll_detector(blend_transient_arrays, PHACT_401.tau_detect)
    assert spikes == [], f"자체 전이에 오검출: {[e.describe() for e in spikes]}"


# ------------------------------------------ CONTINUOUS_OVERLOAD 감지/해제
def _cont_arrays(tau_cmd_level, tau_level, n_sec=1.6, rate=1000):
    n = int(n_sec * rate)
    t = np.arange(n) / rate
    z = np.zeros((n, N))
    tau_cmd = z.copy(); tau_cmd[:, 1] = tau_cmd_level
    tau = z.copy(); tau[:, 1] = tau_level
    return {"t": t, "q": z, "qd": z, "tau": tau, "q_des": z, "tau_cmd": tau_cmd}


def test_overload_fires_on_command_torque_not_net_torque():
    """외란과 맞서는 홀드: 출력측 순토크(tau)는 0인데 지령(tau_cmd)은 7Nm 지속.
    발열은 지령 쪽 — tau_cmd 기준으로 발화해야 한다."""
    det = FailureDetector(N, DetectorConfig(cont_budget=PHACT_401.tau_cont))
    events = det.check(_cont_arrays(tau_cmd_level=7.0, tau_level=0.0))
    assert len(events) == 1
    ev = events[0]
    assert ev.type == FailureType.CONTINUOUS_OVERLOAD and ev.joint_idx == 1
    assert ev.snapshot["tau_avg_1s"] == pytest.approx(7.0, abs=0.1)


def test_overload_not_fired_by_short_burst():
    """정당한 버스트(0.2초 15Nm)는 1초 평균을 못 넘긴다 — 지속 과부하와 구분."""
    a = _cont_arrays(tau_cmd_level=0.5, tau_level=0.0)
    burst = slice(-200, None)                      # 마지막 0.2s만 15Nm
    a["tau_cmd"][burst, 1] = 15.0
    det = FailureDetector(N, DetectorConfig(cont_budget=PHACT_401.tau_cont))
    assert det.check(a) == []                      # 평균 ≈ 0.5*0.8+15*0.2 = 3.4 < 5.76


def test_overload_cleared_event_pairs():
    det = FailureDetector(N, DetectorConfig(cont_budget=PHACT_401.tau_cont))
    assert det.check(_cont_arrays(7.0, 0.0))[0].type == FailureType.CONTINUOUS_OVERLOAD
    # 부하가 빠진 뒤 → 해제 이벤트가 정확히 한 번
    cleared = det.check(_cont_arrays(1.0, 0.0))
    assert [e.type for e in cleared] == [FailureType.OVERLOAD_CLEARED]
    assert det.check(_cont_arrays(1.0, 0.0)) == []   # 반복 발행 없음


def test_overload_channel_disabled_by_default():
    det = FailureDetector(N, DetectorConfig())       # cont_budget=None
    # tau_cmd 50Nm 지속 — 채널이 꺼져 있으면 아무것도 안 나와야 한다
    assert det.check(_cont_arrays(50.0, 0.0)) == []


# --------------------------------- 과부하 → 속도 하향 → 해제 → 복원 (전 구간)
def test_overload_slowdown_and_restore_loop():
    """내려가는 길과 올라오는 길이 모두 있어야 한다 (고전적 함정 방지).

    CONTINUOUS_OVERLOAD → (규칙 폴백) time_scale 0.6 → OVERLOAD_CLEARED →
    (규칙 직행, LLM/쿨다운 우회) time_scale 1.0 복원. 노드 실효 스케일이
    저역통과를 타고 실제로 따라오는지까지 확인.
    """
    stack = FullStack.build(StackConfig())           # client 없음 → 규칙 폴백
    stack.chunk_node.set_active(stack.dictionary.get("direct"), t_now=0.0)
    stack.step(200, monitor=False)

    ev = FailureEvent(FailureType.CONTINUOUS_OVERLOAD, 1, 0.6, t=stack.hal.t,
                      snapshot={"tau_avg_1s": 7.0, "budget": 5.76})
    stack._route(ev)
    assert stack.mgr.get_params("chunk_switch")["time_scale"] == pytest.approx(0.6)
    applied = [a for a in stack.guard.audit if a.applied is not None]
    assert applied and applied[-1].param == "time_scale"

    stack.step(1500, monitor=False)                  # 저역통과 수렴 (τ=0.3s ×5)
    assert stack.chunk_node.effective_time_scale == pytest.approx(0.6, abs=0.02)

    n_llm_before = stack.agent.stats.get("submitted", 0)
    clr = FailureEvent(FailureType.OVERLOAD_CLEARED, 1, 0.0, t=stack.hal.t)
    stack._route(clr)
    assert stack.mgr.get_params("chunk_switch")["time_scale"] == pytest.approx(1.0)
    assert stack.agent.stats.get("submitted", 0) == n_llm_before, \
        "OVERLOAD_CLEARED는 LLM/쿨다운을 우회하고 규칙에 직행해야 한다"
    assert any("restore via rules" in e.text for e in stack.logger.events())

    stack.step(1500, monitor=False)
    assert stack.chunk_node.effective_time_scale == pytest.approx(1.0, abs=0.02)


# ----------------------------------------------------------- time_scale 경계
def _fake_state(q, t):
    q = np.asarray(q, dtype=float)
    return JointState(q=q, qd=np.zeros_like(q), tau_measured=np.zeros_like(q),
                      timestamp=float(t))


def _run_node(node, n_ticks, dt=1e-3, t0=0.0, on_tick=None):
    outs = []
    for k in range(n_ticks):
        t = t0 + k * dt
        if on_tick:
            on_tick(k)
        outs.append(node.update({"state": _fake_state(np.zeros(N), t)}))
    return outs


def _basic_node(scale=None):
    d = default_dictionary()
    node = ChunkSwitchNode(scorer=None, goal=TASK)
    if scale is not None:
        node.params["time_scale"] = scale
        node._scale = scale          # 저역통과 과도 건너뛰기 (수학 검증용)
    node.set_active(d.get("direct"), t_now=0.0)
    return node, d


def test_time_scale_slows_playback_and_scales_velocity():
    """scale 0.5: 같은 틱 수에서 위상은 절반, 출력 속도도 절반 (체인룰)."""
    full, _ = _basic_node(scale=1.0)
    half, _ = _basic_node(scale=0.5)
    out_f = _run_node(full, 1300)     # 첫 틱은 dt=0 → 위상(틱 k) = k·dt
    out_h = _run_node(half, 1300)

    assert full.phase == "DONE"       # 1.2s 청크, 1.3s 재생 → 완료
    assert half.phase == "EXECUTING"  # 위상 0.65 — 절반만 진행
    assert half._phase == pytest.approx(0.5 * full._phase, rel=1e-6)

    # 같은 위상 = 같은 경로점: full 틱 k(위상 k·dt) ↔ half 틱 2k(위상 k·dt)
    assert np.allclose(out_h[1198]["q_des"], out_f[599]["q_des"], atol=1e-9)
    v_peak_f = max(float(np.abs(o["qd_des"]).max()) for o in out_f)
    v_peak_h = max(float(np.abs(o["qd_des"]).max()) for o in out_h)
    assert v_peak_h == pytest.approx(0.5 * v_peak_f, rel=0.02)


def test_time_scale_change_keeps_reference_c1():
    """(경계조건 ①/②) 주행 중 scale 1.0→0.4 스텝 변경: 위상 적분이라 위치는
    정의상 연속, 저역통과 덕에 속도 점프도 가속도×dt 수준이어야 한다."""
    node, _ = _basic_node()
    dt = 1e-3
    prev, max_gap, max_jump = None, 0.0, 0.0

    def on_tick(k):
        if k == 500:
            node.params["time_scale"] = 0.4

    for k in range(2000):
        on_tick(k)
        out = node.update({"state": _fake_state(np.zeros(N), k * dt)})
        if prev is not None:
            pred = prev["q_des"] + prev["qd_des"] * dt
            max_gap = max(max_gap, float(np.abs(out["q_des"] - pred).max()))
            max_jump = max(max_jump, float(np.abs(out["qd_des"] - prev["qd_des"]).max()))
        prev = out

    assert max_gap < 5e-5, f"scale 변경에서 위치 불연속 {max_gap:.2e}"
    assert max_jump < 0.03, f"scale 변경에서 속도 점프 {max_jump:.2e}"
    assert node.effective_time_scale == pytest.approx(0.4, abs=0.02)


def test_switch_during_scaled_playback_keeps_c1():
    """(경계조건 ①) scale 0.6로 재생 중 스위칭: 전이도 같은 위상으로 돌므로
    경계에서 참조가 C1이어야 한다."""
    d = default_dictionary()
    hal = MockRobotHAL(n_joints=N, dt=1e-3, torque_limit=PHACT_401.tau_clamp)
    dream = DreamModel.from_mock_hal(hal, kp=KP, kd=KD)
    dream.torque_limit = np.full(N, PHACT_401.tau_veto)
    node = ChunkSwitchNode(scorer=ChunkScorer(d.all(), dream=dream,
                                              config=ScorerConfig(entry_dir_window_s=0.6)),
                           goal=TASK)
    node.params["time_scale"] = 0.6
    node._scale = 0.6
    node.set_active(d.get("direct"), t_now=0.0)

    dt = 1e-3
    prev, max_gap, max_jump = None, 0.0, 0.0
    for k in range(2500):
        if k == 700:
            dd = np.zeros(N); dd[0] = 5.0
            node.request_switch(dd)
        out = node.update({"state": _fake_state(np.zeros(N), k * dt)})
        if prev is not None:
            pred = prev["q_des"] + prev["qd_des"] * dt
            max_gap = max(max_gap, float(np.abs(out["q_des"] - pred).max()))
            max_jump = max(max_jump, float(np.abs(out["qd_des"] - prev["qd_des"]).max()))
        prev = out

    assert any(x.chosen for x in node.decisions)
    assert max_gap < 5e-5, f"스위칭 경계 위치 불연속 {max_gap:.2e}"
    assert max_jump < 0.05, f"스위칭 경계 속도 점프 {max_jump:.2e}"


def test_stall_not_misdiagnosed_while_crawling():
    """(경계조건 ②) scale 0.3으로 기어가는 동안 STALL이 '느린 진행'을
    고착으로 오진하면 안 된다. (저속 추종은 오차가 작아 구조적으로 안전 —
    이 테스트가 그 논증을 실측으로 고정한다)"""
    stack = FullStack.build(StackConfig())
    stack.mgr.set_params("chunk_switch", {"time_scale": 0.3})
    stack.chunk_node._scale = 0.3
    stack.chunk_node.set_active(stack.dictionary.get("direct"), t_now=0.0)

    stack.step(4500)   # 1.2s 청크를 0.3배속으로 완주 (모니터 on)

    stalls = [e for e in stack.logger.events() if "DETECTED STALL" in e.text]
    assert stalls == [], f"저속 주행을 고착으로 오진: {[s.text for s in stalls]}"
    q = stack.hal.read_state().q
    assert np.abs(q - stack.cfg.goal).max() < 0.05, "기어서라도 도달은 해야 한다"


# ------------------------------------------------------------- 로거 tau_cmd
def test_logger_records_clamped_command_torque():
    """지령이 클램프를 넘으면 잘린 값이 기록돼야 한다 — 잘린 초과분은 실제
    전류로 흐르지 않으므로, 원값을 쓰면 발열 과대평가로 불필요한 slow-down."""
    logger = RingLogger(n_joints=2, torque_clamp=np.array([5.0, 5.0]))
    state = JointState(q=np.zeros(2), qd=np.zeros(2), tau_measured=np.zeros(2),
                       timestamp=0.0)
    cmd = JointCommand(q_des=np.array([10.0, 0.1]), qd_des=np.zeros(2),
                       tau_ff=np.zeros(2), kp=np.full(2, 40.0), kd=np.full(2, 2.0))
    logger.log(state, cmd=cmd)
    tau_cmd = logger.to_arrays()["tau_cmd"][0]
    assert tau_cmd[0] == pytest.approx(5.0)     # 40*10=400 → 클램프 5
    assert tau_cmd[1] == pytest.approx(4.0)     # 40*0.1=4 → 그대로


# --------------------------------------------------------------- AFC 선언
class _HalStub:
    def __init__(self, afc):
        if afc is not None:
            self.afc_state = afc


def test_afc_queried_wins_over_declaration(capsys):
    state, source = resolve_afc_state(_HalStub("off"), declared="on", auto_yes=True)
    assert (state, source) == ("off", "queried")
    assert "경고" in capsys.readouterr().out      # 불일치 경고


def test_afc_declared_when_hal_unknown():
    state, source = resolve_afc_state(_HalStub(None), declared="on", auto_yes=True)
    assert (state, source) == ("on", "declared")


def test_afc_yes_without_declaration_is_an_error():
    """조용히 unknown으로 진행하지 말 것 — 무인 모드에서는 에러로 중단."""
    with pytest.raises(SystemExit, match="--afc"):
        resolve_afc_state(_HalStub("unknown"), declared=None, auto_yes=True)


def test_afc_interactive_prompt_when_unknown():
    answers = iter(["maybe", "ON"])
    state, source = resolve_afc_state(_HalStub(None), declared=None, auto_yes=False,
                                      input_fn=lambda _: next(answers))
    assert (state, source) == ("on", "declared")
