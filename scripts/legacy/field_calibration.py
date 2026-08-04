"""원커맨드 현장 캘리브레이션 파이프라인.

    python scripts/field_calibration.py --hal mock --budget-min 8

단계: 안전 확인 → 여기 궤적 수집 → 품질 리포트 → 모델 2벌 학습 → 비교표 →
모델 선택 → 검증(보정 on/off 추종 오차) → 저장.
각 단계 사이에 계속/중단 확인 (--yes 로 생략). --resume 은 기존 수집 데이터 재사용.

30분 예산 관리를 위해 단계별 소요시간을 출력한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core import MockRobotHAL
from robot_core.legacy.delta import (
    CalibrationData, Collector, DeltaCorrectorNode, MLPDeltaModel,
    PhysicsDeltaModel, SafetyLimits, build_excitation, compare_models,
)
from robot_core.graph import ImpedanceNode, NodeGraphManager
from robot_core.hal import PHACT_401
from robot_core.hal.interface import JointCommand

KP, KD = 40.0, 2.0


# ------------------------------------------------------------- AFC 상태 해석
def resolve_afc_state(hal, declared: str | None, auto_yes: bool,
                      input_fn=input) -> tuple[str, str]:
    """(afc_state, afc_source)를 확정한다. 조용히 unknown으로 진행하지 않는다.

    우선순위: HAL 조회값(queried) > --afc 선언(declared) > 대화형 확인.
    --yes(무인 실행)인데 아무도 모르면 에러 — 모델에 잘못된 AFC 라벨이 붙으면
    나중에 보정기 가드가 속는다.
    """
    hal_state = str(getattr(hal, "afc_state", "unknown"))
    if hal_state in ("on", "off"):
        if declared in ("on", "off") and declared != hal_state:
            print(f"  ** 경고: --afc {declared} 선언했지만 HAL 조회값은 "
                  f"{hal_state} — 조회값을 채택한다 (하드웨어가 정답)")
        return hal_state, "queried"
    if declared in ("on", "off"):
        return declared, "declared"
    if auto_yes:
        raise SystemExit(
            "AFC 상태를 알 수 없다: HAL도 unknown이고 --afc 선언도 없음.\n"
            "--yes(무인) 모드에서는 조용히 unknown으로 진행하지 않는다 — "
            "--afc {on,off} 를 명시할 것.")
    while True:
        ans = input_fn("  AFC(액티브 마찰제거) 상태를 확인해 입력 [on/off]: ").strip().lower()
        if ans in ("on", "off"):
            return ans, "declared"
        print("  'on' 또는 'off'로 입력할 것 (모르면 주최측에 확인).")


# ------------------------------------------------------------------ 유틸
class StepTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.steps: list[tuple[str, float]] = []
        self._last = self.t0

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.steps.append((name, now - self._last))
        self._last = now
        print(f"    ⏱ {name}: {self.steps[-1][1]:.1f}s (누적 {now - self.t0:.1f}s)")

    def report(self) -> str:
        total = time.perf_counter() - self.t0
        lines = ["[단계별 소요시간]"]
        lines += [f"  {n:24s} {d:7.1f}s" for n, d in self.steps]
        lines.append(f"  {'합계':24s} {total:7.1f}s")
        return "\n".join(lines)


def confirm(msg: str, auto_yes: bool) -> bool:
    if auto_yes:
        print(f"{msg} [--yes: 자동 계속]")
        return True
    ans = input(f"{msg} [y/N] ").strip().lower()
    return ans in ("y", "yes")


def make_hal(kind: str):
    if kind == "mock":
        # 리허설용: 마찰/백래시를 '현실적인 미지값'으로 켠 목.
        # 포화는 순간 상한(tau_clamp) — 연속 예산은 별도 감시의 몫이다.
        return MockRobotHAL(n_joints=3, dt=1e-3,
                            torque_limit=PHACT_401.tau_clamp,
                            coulomb_friction=0.4, viscous_friction=0.15,
                            backlash_width=0.02, enable_backlash=True)
    if kind == "real":
        from robot_core.legacy.hal_real import RealRobotHAL
        # TODO(현장): channel/motor_ids 를 실제 값으로
        return RealRobotHAL(channel="can0", motor_ids=[1, 2, 3])
    raise ValueError(f"unknown --hal {kind!r}")


# ------------------------------------------------------------- 검증 실행
def validation_run(hal, model, joints, duration_s: float = 6.0,
                   afc_state: str = "unknown") -> str:
    """테스트 사인 궤적을 보정 off/on으로 추종해 RMS 오차 비교."""
    n = hal.n_joints
    # Δτ 클램프는 지속 예산(tau_cont) 기준 — HAL 한계(순간 상한)가 아니다
    corrector = DeltaCorrectorNode(
        model=model, torque_limits=np.full(n, PHACT_401.tau_cont),
        params={"gain": 1.0, "max_frac": 0.3, "fade_s": 0.0})
    corrector.set_current_afc_state(afc_state)

    def track(correct: bool) -> np.ndarray:
        if hasattr(hal, "reset"):
            hal.reset()
        if hasattr(model, "reset"):
            model.reset()
        corrector.disable_correction()
        if correct and not corrector.enable_correction():
            print(f"  ** {corrector.last_refusal}")
        errs = []
        steps = int(duration_s * 1000)
        for k in range(steps):
            t = k * 1e-3
            q_des = np.zeros(n)
            qd_des = np.zeros(n)
            for j in joints:
                q_des[j] = 0.4 * np.sin(2 * np.pi * 0.5 * t)
                qd_des[j] = 0.4 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
            s = hal.read_state()
            cmd = JointCommand(q_des=q_des, qd_des=qd_des, tau_ff=np.zeros(n),
                               kp=np.full(n, KP), kd=np.full(n, KD))
            out = corrector.update({"state": s, "command": cmd})
            hal.send_command(out["command"])
            errs.append(np.abs(hal.read_state().q - q_des))
        return np.sqrt(np.mean(np.square(errs), axis=0))

    rms_off = track(False)
    rms_on = track(True)
    lines = ["[VALIDATION] 보정 off/on 추종 RMS 오차 [rad]",
             "  jnt |   off    |    on    | 개선"]
    for j in joints:
        imp = (1 - rms_on[j] / rms_off[j]) * 100 if rms_off[j] > 0 else 0.0
        lines.append(f"  {j:3d} | {rms_off[j]:.5f} | {rms_on[j]:.5f} | {imp:+.1f}%")
    lines.append(corrector.timing_report())
    return "\n".join(lines)


# ------------------------------------------------------------------ 메인
def run_pipeline(args) -> dict:
    timer = StepTimer()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "calibration_data.npz"
    joints = [int(j) for j in args.joints.split(",")]

    print("=" * 70)
    print(f"현장 캘리브레이션 시작  hal={args.hal}  budget={args.budget_min}분  "
          f"joints={joints}")
    print("=" * 70)

    hal = make_hal(args.hal)

    # AFC 상태 확정 — 조용히 unknown으로 진행하지 않는다 (모델에 잘못된
    # AFC 라벨이 붙으면 보정기 가드가 속는다)
    afc_state, afc_source = resolve_afc_state(
        hal, getattr(args, "afc", None), args.yes)
    print(f"\nAFC 상태: {afc_state} (출처: {afc_source})")

    # ---------------- 0단계: 안전 확인 ----------------
    print("\n[0/6] 안전 확인")
    print("  - 로봇 주변에 사람/장애물 없음?")
    print("  - E-stop 손 닿는 곳에 있음?")
    print(f"  - 여기 궤적은 관절한계·속도한계의 50% 안에서만 움직임 (약 "
          f"{args.budget_min}분)")
    if not confirm("  위 항목 확인 완료. 여기 궤적을 시작할까?", args.yes):
        print("중단.")
        return {"status": "aborted", "step": "safety"}

    # ---------------- 1단계: 수집 (또는 재사용) ----------------
    if args.resume and data_path.exists():
        print(f"\n[1/6] --resume: 기존 데이터 재사용 ({data_path})")
        data = CalibrationData.load(data_path)
        if data.afc_state != afc_state:
            print(f"  ** 경고: 저장된 데이터의 AFC={data.afc_state}"
                  f"({data.afc_source})인데 현재 확정값은 {afc_state}({afc_source})"
                  f" — AFC 설정이 바뀌었다면 재수집할 것 (--resume 빼고 재실행)")
    else:
        print("\n[1/6] 여기 궤적 실행 + 수집")
        plan = build_excitation(joints=joints, joint_limits=hal.joint_limits,
                                budget_s=args.budget_min * 60.0,
                                n_joints=hal.n_joints)
        print(plan.summary())
        # 중단 기준은 연속 예산 절대값 (HAL 한계는 순간 상한이라 기준이 못 된다)
        collector = Collector(hal, kp=KP, kd=KD, realtime=(args.hal == "real"),
                              safety=SafetyLimits(tau_abs=PHACT_401.tau_cont,
                                                  qd_abs=PHACT_401.qd_limit),
                              on_log=lambda m: print("  [collector]", m))
        data = collector.collect(plan)
        # HAL이 못 준 AFC 상태를 확정값으로 채운다 (출처 포함)
        data.afc_state, data.afc_source = afc_state, afc_source
        data.save(data_path)
        print(f"  저장: {data_path}")
    timer.mark("수집")

    # ---------------- 2단계: 품질 리포트 ----------------
    print("\n[2/6] 수집 품질")
    print(data.quality_report())
    if data.aborted:
        print("  ** 수집이 안전 중단됨 — 부분 데이터로 진행할지 판단할 것")
    if not confirm("  품질 OK. 모델 학습으로 진행?", args.yes):
        return {"status": "aborted", "step": "quality", "data": data}

    # ---------------- 3단계: 모델 2벌 학습 ----------------
    print("\n[3/6] 물리 모델 피팅")
    physics = PhysicsDeltaModel(hal.n_joints).fit(data, joints)
    print(physics.info())
    sanity = physics.sanity_warnings()
    if sanity:
        print("  ** 피팅 sanity 경고 (AFC 상태 기준):")
        for w in sanity:
            print(f"     - {w}")
        if not confirm("  경고를 확인했다. 그래도 진행?", args.yes):
            return {"status": "aborted", "step": "fit_sanity", "data": data}
    else:
        print(f"  sanity OK (AFC={physics.afc_state} 기준 판정)")
    timer.mark("물리 피팅")

    mlp = None
    try:
        print("\n      MLP 학습 (CPU, 최대 3분)")
        inertia = np.array([physics.params[j].inertia for j in range(hal.n_joints)])
        mlp = MLPDeltaModel(hal.n_joints).fit(
            data, joints, inertia=inertia, seed=args.seed,
            on_log=lambda m: print("  [mlp]", m))
    except ImportError as e:
        print(f"  MLP 생략: {e}")
    timer.mark("MLP 학습")

    # ---------------- 4단계: 비교표 + 선택 ----------------
    print("\n[4/6] 모델 비교")
    table, result = compare_models(data, physics, mlp, joints)
    print(table)

    if args.yes or mlp is None:
        choice = result["overall"] if mlp is not None else "physics"
        print(f"  자동 선택: {choice}")
    else:
        ans = input("  사용할 모델 [physics/mlp] (기본 physics): ").strip().lower()
        choice = "mlp" if ans == "mlp" and mlp is not None else "physics"
    model = mlp if choice == "mlp" else physics
    timer.mark("비교/선택")

    # ---------------- 5단계: 검증 ----------------
    print(f"\n[5/6] 검증 실행 ({choice} 모델, 보정 off vs on)")
    if not confirm("  로봇이 검증 궤적을 추종한다. 시작?", args.yes):
        return {"status": "aborted", "step": "validation"}
    val_report = validation_run(hal, model, joints, afc_state=afc_state)
    print(val_report)
    timer.mark("검증")

    # ---------------- 6단계: 저장 ----------------
    print("\n[6/6] 저장")
    physics_path = physics.save(out_dir / "physics_model.npz")
    print(f"  물리 모델: {physics_path}")
    mlp_path = None
    if mlp is not None:
        mlp_path = mlp.save(out_dir / "mlp_model.npz")
        print(f"  MLP 모델 : {mlp_path}")
    (out_dir / "report.txt").write_text(
        "\n\n".join([data.quality_report(), physics.info(), table, val_report,
                     timer.report()]), encoding="utf-8")
    print(f"  리포트   : {out_dir / 'report.txt'}")
    print(f"  선택 모델: {choice}")
    timer.mark("저장")

    print()
    print(timer.report())
    return {"status": "ok", "choice": choice, "data": data, "physics": physics,
            "mlp": mlp, "validation": val_report, "out_dir": out_dir}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hal", choices=["mock", "real"], default="mock")
    p.add_argument("--budget-min", type=float, default=8.0,
                   help="여기 궤적 총 소요시간 [분]")
    p.add_argument("--joints", default="0,1,2", help="캘리브레이션할 관절 (쉼표 구분)")
    p.add_argument("--out-dir", default="calib_out")
    p.add_argument("--resume", action="store_true", help="기존 수집 데이터 재사용")
    p.add_argument("--afc", choices=["on", "off", "unknown"], default=None,
                   help="AFC 상태 선언 (HAL이 unknown일 때 채택, 출처=declared)")
    p.add_argument("--yes", action="store_true", help="모든 확인 프롬프트 자동 통과")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    result = run_pipeline(args)
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
