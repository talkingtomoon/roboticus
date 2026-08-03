# robot_core — 해커톤 코어 인프라

하드웨어와 미션이 무엇이든 갈아끼울 수 있게 만든 최소 뼈대.
**실물 로봇 없이 `pytest`만으로 전부 검증된다.** 캠프 현장에서는
[`robot_core/hal/real.py`](robot_core/hal/real.py) 한 파일만 채우면 나머지는 그대로 돌아간다.

ROS 2 / Isaac Sim 의존성 없음. 순수 파이썬 + numpy.

```
robot_core/
  hal/        interface.py(ABC) · mock.py(시뮬) · real.py(현장에서 채울 stub)
  control/    runner.py — 1kHz 루프, 주파수·지터·오버런 측정
  logging/    ring_logger.py — 최근 N초 링버퍼 + LLM에 먹일 텍스트 덤프
  graph/      노드 그래프 매니저 (DAG·위상 정렬·YAML) + 표준 제어 노드
  recovery/   실패 감지기 · SafetyGuard · 규칙 폴백 · LLM 에이전트 · 슈퍼바이저
  chunks/     모션 청크 (큐빅 스플라인) + 생성기 (min-jerk·우회·후퇴·변주)
  switching/  실시간 채점기(Dream 롤아웃) · C1 블렌더 · 스위칭 매니저 노드
  delta/      잔차 보정: 여기 궤적 · 수집기 · 모델 2벌(물리/MLP) · 보정 노드
  integration/ 세 지름길 통합: full_stack(표준 조립+인터록) · scenarios(5종) · timeline
  adapters/   ros2_adapter.py — rclpy 래핑법 주석 스켈레톤 (현장에서 채움)
scripts/      field_calibration.py · baseline_switch_comparison.py ·
              check_llm_model.py
tests/        251개 테스트, 실물·실제 API 없이 전부 통과
examples/     demo_impact_and_jam.py · demo_self_recovery.py ·
              demo_chunk_switching.py · demo_full_rehearsal.py
```

## 빠른 시작

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install numpy pyyaml pytest
python -m pytest -q
python examples/demo_impact_and_jam.py
python examples/demo_self_recovery.py --no-llm   # 자가 회복 루프 (규칙 폴백 경로)
python examples/demo_chunk_switching.py          # 충격 → 우회 궤적 스위칭
```

실제 LLM 경로를 쓰려면 `pip install anthropic` 후 `ANTHROPIC_API_KEY` 환경변수를 설정한다.
**키가 없어도 모든 것이 동작한다** — 시뮬레이트된 LLM 또는 규칙 폴백으로 간다.

`pip install -e .` 도 되지만 필수는 아니다 (저장소 루트에서 실행하면 import 된다).

## 핵심 규약: 임피던스 제어

모든 QDD 액추에이터 지령은 이 한 줄로 표현된다. HAL 구현체는 이걸 지켜야 한다.

```
tau = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff
```

```python
import numpy as np
from robot_core import MockRobotHAL, JointCommand, ControlLoopRunner, RingLogger

hal = MockRobotHAL(n_joints=4, dt=1e-3)
logger = RingLogger.for_hal(hal, window_sec=3.0, threshold_frac=0.6)
runner = ControlLoopRunner(hal, rate_hz=1000.0, logger=logger)

def policy(state):                       # fn(state) -> command
    n = hal.n_joints
    return JointCommand(
        q_des=np.full(n, 0.3), qd_des=np.zeros(n), tau_ff=np.zeros(n),
        kp=np.full(n, 50.0), kd=np.full(n, 2.0),
    )

runner.set_policy(policy)
stats = runner.run(duration_sec=1.0)
print(stats.summary())        # 실제 달성 Hz / 지터 / 오버런
print(logger.dump_text())     # 사람·LLM이 읽는 요약
```

알고리즘 모듈은 **`RobotHAL` 인터페이스만 보고** 개발할 것. `MockRobotHAL`이든
`RealRobotHAL`이든 바꿔 끼우기만 하면 되게 유지한다.

## 구성 요소

### HAL (`robot_core/hal/`)

| 타입 | 내용 |
|---|---|
| `JointState` | `q`, `qd`, `tau_measured` (전부 `np.ndarray`, shape=`(n,)`), `timestamp` |
| `JointCommand` | `q_des`, `qd_des`, `tau_ff`, `kp`, `kd` — 헬퍼: `.hold(q)`, `.damping_only(n)` |
| `RobotHAL` | ABC: `read_state()`, `send_command()`, `n_joints`, `joint_limits`, `torque_limits` |

`joint_limits`/`torque_limits`는 항상 복사본을 돌려준다 — 상위 코드가 실수로 덮어써도 HAL 내부는 안 망가진다.

### MockRobotHAL

관절별 독립 1자유도 근사(`M*qdd + C*qd + G = tau`). 다관절 커플링은 일부러 무시했다 —
목적은 로직 검증이지 물리 정확도가 아니다.

전부 생성자에서 켜고 끌 수 있다:

```python
MockRobotHAL(
    n_joints=6, dt=1e-3,
    inertia=0.05, viscous_friction=0.02, coulomb_friction=0.1,
    backlash_width=0.01, gravity_torque=0.0,
    torque_limit=30.0, joint_limit=(-3.14, 3.14),
    enable_viscous_friction=True, enable_coulomb_friction=True,
    enable_backlash=False, enable_saturation=True,
)
```

**고장/외란 주입 API** — 상위 감지 로직을 실물 없이 시험하는 용도:

```python
hal.inject_disturbance(joint_idx=2, torque=20.0, duration=0.03)  # 충격 → tau_measured가 튄다
hal.inject_jam(0)          # 관절 고착 → 위치 고정, PD 오차 누적으로 토크 포화
hal.clear_jam(0)           # 해제 (인자 없으면 전부)
hal.reset()                # 상태·주입 전체 초기화
```

`tau_measured`는 **관절에 실제로 걸린 구동 토크 + 외란**으로 모델링했다.
접촉/충격이 측정 토크에 그대로 보이게 하려는 의도적 선택이다.

주의: 이미 목표에 도달해 정지한 관절을 jam시키면 PD 오차가 안 생겨서 토크도 안 튄다.
**jam을 감지하려면 걸린 관절에 계속 새 목표를 줘야 한다** (데모 3단계 참고). 이건 실물도 똑같다.

Coulomb 마찰(기본 0.1 Nm)이 켜져 있으면 정상상태 위치 오차가 0으로 수렴하지 않고
`kp=40` 기준 ~1e-3 rad 바닥값이 남는다. 버그가 아니라 스틱 구간이다.

### ControlLoopRunner

```python
runner = ControlLoopRunner(hal, rate_hz=1000.0, logger=logger,
                           spin_wait=True, on_error="raise")
runner.set_policy(fn)              # fn(state) -> JointCommand
stats = runner.run(duration_sec=1.0)   # 또는 n_steps=1000
runner.stop()                      # 정책 콜백 안에서도 호출 가능
```

- `LoopStats`: 달성 Hz, 주기 mean/std(지터)/min/p50/p95/p99/max, 계산시간 mean/p50/p95/max, 오버런 수, 정책 예외 수. `stats.summary()`로 한 번에 출력.
- **오버런**: 주기 안에 일을 못 끝낸 횟수. 밀렸을 때 따라잡지 않고 마감을 현재 시각으로 리셋한다(한 번 밀렸다고 몰아치지 않게).
- `spin_wait=True`는 마감 1.5ms 전까지만 자고 나머지를 busy-wait으로 채운다. 윈도우에서 1kHz를 맞추려면 사실상 필수 (측정치: 1000.4Hz, 지터 0.11ms).
- 정책이 예외를 던지면 `on_error="raise"`는 **감쇠 지령을 먼저 보내고** 예외를 올린다 (하드웨어를 그냥 놓지 않는다). `"hold"`는 직전 지령 유지하며 루프를 계속하고 카운트만 올린다.
- 데스크톱 OS는 하드 실시간이 아니다. `compute_max`는 OS 선점 한 번에 크게 튀니 진단은 **p95**를 볼 것.

### RingLogger

최근 `window_sec` 구간만 메모리에 유지한다. `dump_text()`는 같은 입력이면 같은 출력이다
(본문에 벽시계 시각을 안 넣는다 → 스냅샷 테스트 가능).

```python
logger = RingLogger.for_hal(hal, window_sec=3.0, threshold_frac=0.6)
logger.mark_event("jam on joint 0", t=hal.t)   # 타임라인 메모
print(logger.dump_text(window_sec=0.6))
arrays = logger.to_arrays()                     # 분석용 원본 numpy 배열
```

덤프 섹션: `[TIMING]`(달성 Hz·지터·오버런) / `[PER-JOINT TORQUE]`(mean·rms·peak·임계 초과) /
`[PER-JOINT TRACKING]`(q vs q_des 오차) / `[THRESHOLD EXCURSIONS]`(초과 구간 시작·끝·피크, 진행 중이면 `[ONGOING]`) /
`[EVENTS]` / `[LAST SAMPLE]`. 실제 출력 예시는 [tests/snapshots/dump_text.txt](tests/snapshots/dump_text.txt) 참고.

로그가 섞여 시계가 뒤로 가면 엉터리 통계 대신 `not monotonic` 이라고 말한다.

---

## 캠프 현장 체크리스트 — `RealRobotHAL` 채우는 법

파일: [`robot_core/hal/real.py`](robot_core/hal/real.py). 각 메서드에 같은 내용이 TODO로 달려 있다.
**순서대로** 할 것. 특히 6번을 건너뛰지 말 것.

### 0. 받자마자 확인 (코드 짜기 전)
- [ ] CAN 인터페이스 이름 (`can0` / `slcan0` / `PCAN_USBBUS1`) 과 비트레이트 (보통 1 Mbps)
- [ ] 드라이버 프로토콜: MIT/Mini-Cheetah 계열? CyberGear? ODrive? 벤더 SDK?
- [ ] 모터 CAN ID 목록과 **관절 순서** (이 순서가 곧 배열 인덱스다)
- [ ] 기어비, 관절별 정격/최대 토크, 기구학적 관절 한계
- [ ] 드라이버 자체 워치독이 있는지 (프레임 끊기면 스스로 끄는지)

### 1. 통신 (`connect`)
- [ ] `can.interface.Bus(channel=..., bustype=..., bitrate=...)` 또는 벤더 SDK 핸들을 `self._bus`에
- [ ] `__init__`에서 `self._n`, `self._joint_limits`, `self._torque_limits`를 **실제 값**으로 채우고
      `joint_limits` / `torque_limits` 프로퍼티의 `NotImplementedError`를 지운다
- [ ] 토크 한계는 데이터시트 값 그대로 쓰지 말고 **정격의 60~70%**로 시작할 것

### 2. 인에이블 (`enable`)
- [ ] 드라이버별 enable 프레임 전송 (MIT 계열이면 보통 `0xFF..FC`=enable, `FD`=disable, `FE`=set zero)
- [ ] **인에이블 직후 첫 지령은 반드시 `kp=kd=0, tau_ff=0`.** 안 그러면 첫 프레임에서 로봇이 튄다

### 3. 상태 읽기 (`read_state`)
- [ ] 피드백 프레임에서 `(q, qd, tau)` 언팩 — 고정소수점이면 `_unpack_float` 헬퍼 사용 (테스트 완료)
- [ ] 관절축 환산: `q = direction * (q_motor / gear_ratio) - offset`
- [ ] 프레임 유실 시 **예외 대신 직전 값 유지 + 카운터 증가**. 1kHz 루프에서 예외를 던지면 루프가 죽는다

### 4. 지령 보내기 (`send_command`)
- [ ] **전송 직전 클램프** — 이게 상위 로직 버그로부터 하드웨어를 지키는 마지막 방어선이다
      ```python
      q_des  = np.clip(cmd.q_des, self._joint_limits[:,0], self._joint_limits[:,1])
      tau_ff = np.clip(cmd.tau_ff, -self._torque_limits, self._torque_limits)
      kp, kd = np.clip(cmd.kp, 0, KP_MAX), np.clip(cmd.kd, 0, KD_MAX)
      ```
- [ ] 관절축 → 모터축 역변환, 게인 단위 확인 (기어비 있으면 `kp_motor = kp_joint / ratio**2` 같은 환산 필요)
- [ ] 버스 에러 카운트, 연속 N회 실패 시 `estop()`

### 5. 안전 (`estop`, `disable`, `close`)
- [ ] `estop()`: 게인 0 프레임 전 관절 전송 → `disable()`
- [ ] `with RealRobotHAL(...) as hal:` 컨텍스트 매니저는 이미 구현돼 있다 —
      예외로 빠져나가도 `estop()` 후 `close()`가 보장된다
- [ ] 드라이버 워치독이 없으면 별도 스레드로 타임아웃 감시 추가

### 6. 검증 — 전체 인에이블 전에 반드시
- [ ] **모터 1개만** 인에이블, `kp=0, kd=0.5` (댐핑만) → 손으로 돌려보며 `q` 부호 확인
- [ ] 부호가 반대면 `joint_direction`만 뒤집는다 (다른 코드 고치지 말 것)
- [ ] `kp=1~2`로 아주 약하게, `q_des = 현재 q` 홀드 테스트
- [ ] 그 다음에야 나머지 관절 인에이블

### 7. 스모크 테스트
- [ ] `tests/test_hal_mock.py`의 PD 추종 테스트를 `RealRobotHAL`로 돌리되
      **관절 하나만, kp 낮게, 진폭 작게** 바꿔서
- [ ] `ControlLoopRunner`로 1kHz 실행 → `stats.summary()`의 오버런과 p95 확인
- [ ] `RingLogger.dump_text()` 출력이 실물에서도 말이 되는지 눈으로 확인
- [ ] ⚠️ **외란/jam 주입 API는 목 전용이다.** 실물에서 실패 감지를 시험하려면
      사람이 직접 관절을 막거나 밀어야 한다

---

## 노드 그래프 (`robot_core/graph/`)

제어 모듈을 노드로 등록하고 런타임에 켜고 끄는 구조. ROS 2 비의존 —
현장에서 ROS 2 위에 올리는 법은 [`adapters/ros2_adapter.py`](robot_core/adapters/ros2_adapter.py)
주석 스켈레톤 참고 (우리 노드 인터페이스가 rclpy 노드에 1:1 대응되게 설계돼 있다).

```python
from robot_core.graph import NodeGraphManager, TargetNode, ImpedanceNode, STANDARD_NODE_TYPES

mgr = NodeGraphManager()
mgr.add_node(TargetNode(params={"depth": 0.4}))       # q_des 생성
mgr.add_node(ImpedanceNode(params={"kp": 25.0}))       # JointCommand 생성
mgr.connect("target", "impedance")                     # DAG 간선 (사이클은 즉시 에러)

# 제어 루프 정책 = 그래프 1스텝
runner.set_policy(lambda st: mgr.step({"state": st})["impedance"]["command"])

# 런타임 API (전부 스레드 안전)
mgr.enable("target"); mgr.disable("target")
mgr.set_params("impedance", {"kp": 60.0})   # 없는 키면 KeyError (오타 방지)
mgr.get_graph_spec()

# 현장에서 코드 안 고치고 위상 변경
mgr.save_yaml("graph.yaml")
mgr = NodeGraphManager.from_yaml("graph.yaml", STANDARD_NODE_TYPES)
```

새 노드는 `Node`를 상속해 `update(inputs) -> dict` 하나만 구현한다.
**LLM이 조정할 파라미터는 반드시 float 스칼라로 둘 것** (SafetyGuard가 float만 다룬다).

## 자가 회복 루프 (`robot_core/recovery/`)

```
1kHz 제어 루프 ──▶ RingLogger ──▶ [모니터 20Hz] FailureDetector
   ▲                                    │ FailureEvent
   │ set_params (스레드 안전)            ▼
NodeGraphManager ◀── SafetyGuard ◀── LLMRecoveryAgent (워커 스레드)
                       (관문)              │ 실패 시 ▼
                                       RuleBasedRecovery (폴백)
```

절대 원칙 세 가지가 코드 구조로 강제돼 있다:

1. **LLM은 제어 루프를 절대 블로킹하지 않는다.** 호출은 전부 워커 스레드 안.
   `submit()`은 큐에 넣고 즉시 리턴. 테스트로 검증됨 (LLM이 1초 걸려도 루프 500Hz 유지).
2. **LLM 출력을 그대로 믿지 않는다.** 모든 액션은 SafetyGuard를 통과해야만 적용.
   화이트리스트 밖 → 거부, 범위 밖 → 클램프, 급변 → 변화율 제한, 전 이력 감사 로그.
   파싱 실패·타임아웃(3s)·낮은 신뢰도 → 규칙 폴백.
3. **LLM 없이도 동작한다.** `client=None`이면 곧장 `RuleBasedRecovery`로 간다.
   데모의 `--no-llm` 플래그가 이 경로다.

### 실패 감지기 — 왜 세 종류인가

**토크 임계치만으로는 절반을 놓친다** (1단계에서 확인: 목표에 도달해 정지한
관절이 걸리면 PD 오차가 없어 토크가 안 튄다). 그래서:

| 타입 | 조건 | 잡는 상황 |
|---|---|---|
| `TORQUE_SPIKE` | \|tau\| > 임계치가 20ms+ 지속 | 움직이다 걸림 |
| `STALL` | 지령-실제 오차 큰데 \|qd\|≈0이 250ms+ | **정지 중 고착** (토크 감지 불가 영역) |
| `OSCILLATION` | 유의미한 진폭의 qd 부호 반전 8회/s+ | 게인 과다 |

전 감지기에 최소 지속시간 + 히스테리시스 + refractory가 걸려 있다 (한 샘플 튐으로 발동 금지).
판정은 `RingLogger.to_arrays()` 원본 배열로만 한다 — `dump_text()` 포맷에 비의존.

### 조정 가능 파라미터 화이트리스트를 미션에 맞게 수정하는 법

화이트리스트는 **코드 어디에도 흩어져 있지 않고 SafetyGuard 생성자 한 곳**에 모인다.
미션이 공개되면 이것만 고치면 된다:

```python
from robot_core.recovery import SafetyGuard, ParamSpec

guard = SafetyGuard(mgr, {
    #  "노드.파라미터":  ParamSpec(min,  max,  max_rel_step, max_abs_step)
    "impedance.kp":      ParamSpec(1.0, 120.0, max_rel_step=2.5),
    "impedance.kd":      ParamSpec(0.1,   8.0, max_rel_step=2.5),
    "target.depth":      ParamSpec(0.0,   1.0, max_rel_step=None, max_abs_step=0.2),
    "target.retreat":    ParamSpec(0.0,   0.3, max_rel_step=None, max_abs_step=0.1),
})
```

수정 절차:
1. **노드부터.** 미션용 노드를 만들고 조정할 값을 `params`의 float로 노출한다.
2. **범위는 하드웨어 한계보다 안쪽으로.** `max`는 "LLM이 이 값을 골라도 로봇이
   안 부서지는" 값이지 데이터시트 한계가 아니다. 예: kp 상한은
   `토크한계 / 최대예상오차`보다 작게 (kp 120 × 오차 0.17rad ≈ 토크한계 20Nm).
3. **변화율 제한 선택.** 배율형 파라미터(게인)는 `max_rel_step`(2~3배),
   덧셈형(오프셋·거리)은 `max_abs_step`. 현재값이 0이 될 수 있는 파라미터는
   배율 제한이 무의미하므로 **반드시 `max_abs_step`을 줄 것**.
4. **여기 없는 건 LLM이 절대 못 건드린다.** 프롬프트의 허용 목록도, 검증도
   전부 이 dict 하나에서 나온다 (`guard.describe_whitelist()`).
5. 규칙 폴백(`RuleBasedRecovery`)의 기본 전략은 `default_rules()`에 있다 —
   노드 이름이 다르면 `default_rules(impedance_node=..., target_node=...)`로 맞추거나
   `RuleAction` 리스트를 직접 넘긴다. 규칙이 낸 값도 어차피 SafetyGuard를 다시 통과한다.

LLM 비용/거동 설정은 전부 `LLMConfig`에 있다 (모델, max_tokens, effort,
타임아웃 3s, 쿨다운, 최소 신뢰도, 프롬프트 로그 창). API 키는 환경변수
`ANTHROPIC_API_KEY`로만 받는다 — 코드에 넣지 말 것.

### 데모

```bash
python examples/demo_self_recovery.py           # 키 있으면 실제 LLM, 없으면 시뮬레이트
python examples/demo_self_recovery.py --no-llm  # 규칙 폴백 경로
```

타임라인 출력: 정상 접근 → 저항 외란(-8Nm)으로 STALL 유발 → 감지 → LLM/규칙이
kp 상향 → 오차 0.33→0.13rad 회복 → jam으로 TORQUE_SPIKE 유발 → 후퇴+kp 하향 →
감사 로그와 dump_text로 전 과정 확인.

## 모션 청크 스위칭 (`robot_core/chunks/` + `robot_core/switching/`)

"로봇이 이동 중 밀리면, 후보 궤적들을 실시간 채점해서 가장 순응적인 궤적으로
부드럽게 갈아탄다" — DREAM-Chunk의 단순화 재해석.

```
1kHz 제어 루프 ──▶ ChunkSwitchNode ──(q_des, qd_des)──▶ ImpedanceNode
                      ▲ request_switch(외란벡터)          [상태기계: EXECUTING→SCORING→BLENDING]
모니터 (10~50ms) ──┘
  FailureDetector(재사용) + estimate_disturbance()
```

- **채점 (scorer.py)**: 연결 비용(현재 상태↔진입점 거리) + 저항 비용((1-cos)/2 —
  순응 0 < 비켜감 0.5 < 정면 대항 1) + 진행 비용(목표까지 남는 거리)의 가중합.
  전 과정 numpy 벡터화 — **후보 100개 채점 실측 ~1-2ms** (예산 5ms, 테스트로 강제).
- **Dream 요소**: 각 후보의 첫 0.15s를 관절별 1자유도 모델(DreamModel, 주입식)로
  순전파해 예상 토크 지령을 계산, 한계 초과가 예측되는 후보는 **탈락(veto)**.
  전부 탈락하면 스위칭하지 않고 사유를 기록한다 (참조가 멀리 전진한 뒤의 늦은
  요청은 전부 veto되는 게 정상 — 그래서 감지 즉시 요청하는 게 중요하다).
- **블렌딩 (blender.py)**: 현재 참조 상태에서 새 청크로 양끝 (위치·속도·가속)
  경계조건을 만족하는 5차 다항식 전이. 전이 시간은 상태 거리에 비례(최소 0.15s).
  블렌더는 무상태라 전이 중 재스위칭도 재귀적으로 안전하다.
  실측 연속성: 스위칭 순간 포함 위치 갭 1e-5 rad, 속도 점프 0.02 rad/s(=qdd·dt) 수준.
- **감사 로그**: 모든 결정(기각 포함)에 후보별 점수표가 남는다 —
  `node.dump_decisions()`가 "왜 이 궤적을 골랐는지" 데모 자료가 된다.
- 스위칭 판단에 LLM 없음 — 이건 ms 단위 반응 경로다. LLM은 지름길 ①의
  초 단위 회복 경로에만 산다.

### 현장에서 미션 보고 딕셔너리 채우는 절차

미공개 미션 대비: 청크의 "내용"이 아니라 "형식과 파이프라인"이 완성돼 있다.
캠프에서 미션이 공개되면 아래 순서로 내용물만 채운다:

```python
import numpy as np
from robot_core.chunks import ChunkDictionary, generator as G

# 0) 한계를 먼저 정의 — 모든 생성 함수에 넘기면 위반 시 그 자리에서 에러
limits = dict(joint_limits=hal.joint_limits, qd_max=8.0)

# 1) 기본 접근 궤적 (미션의 주 동작)
direct = G.min_jerk("approach", q_home, q_task, duration=1.5,
                    tags=["approach"], **limits)

# 2) 경유점 궤적 (장애물 위치를 알면)
over = G.via_points("approach_over", q_home, [q_via1, q_via2], q_task, 2.0, **limits)

# 3) 변주로 후보군 불리기 — 같은 목표, 다른 회피 방향/속도
d = ChunkDictionary([direct, over])
for axis, sign in [(1, +1), (1, -1), (2, +1)]:          # 회피 방향 후보
    off = np.zeros(n); off[axis] = sign * 0.3
    d.add(G.with_detour(direct, t_via=0.7, offset=off,
                        name=f"detour_j{axis}{'p' if sign > 0 else 'n'}", **limits))
d.add(G.time_scaled(direct, 1.5, name="approach_slow"))  # 느린 변주 (저충격)
d.add(G.retreat("retreat", q_task, q_home, 1.0, **limits))  # 안전 복귀

# 4) 저장 — 딕셔너리 전체가 디렉토리 하나 (청크당 npz 파일 하나)
d.save_dir("chunk_dict/")
# 이후 로드: d = ChunkDictionary.load_dir("chunk_dict/")
# MuJoCo 등 외부 궤적: waypoint로 뽑아 MotionChunk.from_waypoints()로 피팅

# 5) 채점기/노드에 연결
dream = DreamModel.from_mock_hal(hal, kp=KP, kd=KD)   # 실물이면 추정 파라미터로 교체
scorer = ChunkScorer(d.all(), dream=dream,
                     config=ScorerConfig(entry_dir_window_s=0.6))
node = ChunkSwitchNode(scorer=scorer, goal=q_task)
node.set_active(d.get("approach"), t_now=state.timestamp)
```

체크포인트:
- `entry_dir_window_s`는 우회 범프가 구별되는 길이로 (범프 시작 시점보다 길게).
- 후보 수가 늘면 `python -m pytest tests/test_switching.py -k budget`로 5ms 예산 재확인.
- 외란 방향 추정(`estimate_disturbance`)의 `stiffness`에는 실제 kp를 넣을 것.
- 감지 즉시 `request_switch()`를 부를 것 — 늦으면 후보들이 전부 veto된다 (정상).

## 잔차(Delta) 보정 (`robot_core/delta/` + `scripts/field_calibration.py`)

실물의 마찰/백래시는 실물 데이터로만 학습할 수 있다. 그래서 이 모듈의 산출물은
학습된 모델이 아니라 **"현장 30분 안에 수집→학습→검증→적용을 끝내는 파이프라인"**
이고, 목 리허설(`tests/test_field_rehearsal.py`)로 전체가 검증돼 있다.

```
여기 궤적(excitation) → 수집기(안전 감시) → 모델 2벌 학습 → 비교표 → 보정 노드
   사인스윕/반전/램프      npz + 품질 리포트    물리 vs MLP      val RMS     tau_ff += Δτ
```

- **물리 모델**: `tau_c*sign(qd) + b*qd + 백래시 킥`. 강건 최소제곱(잔차 트리밍).
  목 리허설에서 심어둔 파라미터를 **오차 ~0%로 복원** (요구사항 10%).
- **MLP**: 관절별 4→32→32→1, PyTorch CPU 학습(3분 예산) + **numpy 블록행렬 추론**
  (런타임에 torch 불필요). torch 없으면 자동으로 물리 모델만 쓴다.
- **보정 노드 안전장치 3종**: 하드 클램프(**연속 예산 tau_cont**의 ±30% — Δτ는
  지속 신호라 버스트 대역이 아닌 연속 예산 안에서 살아야 한다. nominal 비례가 아닌 이유:
  정지 마찰 돌파 순간 nominal≈0이라 비례 클램프는 정확히 그때 보정을 죽인다),
  페이드인(기본 2초 0→100%), 킬스위치(`disable_correction()` 즉시 Δτ=0).
- 추론 실측: 물리 ~5µs/관절, MLP ~17µs/관절 (예산 50µs, 테스트로 강제).
- LLM 회복 루프 연동: `"delta_corrector.gain": ParamSpec(0.0, 1.0, max_rel_step=None,
  max_abs_step=0.5)`를 SafetyGuard 화이트리스트에 넣으면 지름길 ①이 보정 강도를
  낮출 수 있다 (테스트로 검증됨).
- **사전 학습 금지**: 모델 파일은 항상 현장 데이터에서만 나온다. 저장소에 모델
  파일을 커밋하지 말 것 (목 파라미터에 과적합된 모델을 실물에 올리는 사고 방지).

### 현장 1일차 캘리브레이션 절차서 (30분 예산)

**사전 조건**: RealRobotHAL 구현 완료(위 체크리스트), 로봇이 안전 자세로 정지,
E-stop 확보. `scripts/field_calibration.py`의 `make_hal("real")`에 채널/모터ID 기입.

```bash
# 리허설 (로봇 없이, 언제든): 전체 파이프라인이 지금도 도는지 30초 확인
python -m pytest tests/test_field_rehearsal.py -v

# 본 실행
python scripts/field_calibration.py --hal real --budget-min 8
```

| 단계 | 화면에 보이는 것 | 계속 조건 | 중단 조건 |
|---|---|---|---|
| 0. 안전 확인 | 체크리스트 프롬프트 | 주변 클리어 + E-stop 확보 | 하나라도 아니오 |
| 1. 여기 궤적 (약 8분) | 계획 요약(관절별 피크 위치/속도) | 피크가 한계 50% 안 | 로봇 소리/움직임 이상 → Ctrl-C (부분 데이터 보존됨) |
| 2. 품질 리포트 (30초) | 속도 구간별 샘플 수, 반전 횟수 | "커버리지 양호" 또는 경고 없음 | "저속 샘플 부족"/"반전 부족" 경고 → `--budget-min` 늘려 재수집 |
| 3. 학습 (물리 <1분, MLP <3분) | 물리 파라미터 값 + sanity 판정 | sanity OK. **AFC on이면 tau_c가 0 근처(살짝 음수 포함)인 것이 정상** | AFC off/unknown인데 음수 마찰, 또는 수백 Nm → 데이터 이상, 재수집 (판정은 스크립트가 AFC 상태 기준으로 자동 출력) |
| 4. 비교표 | 무보정/물리/MLP 3열 RMS | — | — |
| 5. 모델 선택 | 프롬프트 | **MLP가 물리보다 15%+ 나을 때만 MLP.** 아니면 물리 (디버깅 가능한 쪽) | |
| 6. 검증 (10초) | 보정 off/on 추종 오차 비교 | 개선율 전 관절 양수 | 악화되는 관절 있으면 그 관절 gain=0으로 시작 |
| 7. 적용 | 저장 경로 | 보정 노드에 `set_model()` + `enable_correction()` (페이드인 2초) | 이상 시 `disable_correction()` = 킬스위치 |

수집만 되어 있으면 학습 재시도는 `--resume`으로 즉시:
```bash
python scripts/field_calibration.py --hal real --resume --yes
```

30분 예산 배분 가이드: 수집 8분 + 품질 판단 1분 + 학습 4분 + 검증 2분 = 15분.
나머지 15분은 재수집 1회분 여유다. 시간이 없으면 `--joints`로 중요 관절만.

## 통합: 세 지름길을 한 그래프에 (`robot_core/integration/`)

```
target ──▶ chunk_switch ──▶ impedance ──▶ delta_corrector ──▶ HAL
             (참조 덮어쓰기)   (nominal 계산)   (tau_ff 가산, 최종 방어선)
                  ▲
        [모니터 20ms 주기] FailureDetector → 인터록 라우팅:
          TORQUE_SPIKE → 스위칭 (처짐 줄어드는 중이면 '해제 잔향'으로 무시)
          STALL        → 회복 루프 (인터록 예외 — 스위칭으로 해결 안 됨)
          그 외        → 회복 루프 (BLENDING 중엔 보류, 끝나면 방출)
```

노드들은 서로를 import하지 않는다 — 라우팅과 인터록은 `FullStack`이 그래프
매니저를 통해서만 수행한다. 감사 로그 3개(감지/스위칭/SafetyGuard)는 전부
**로봇 클록** 기준이라 `timeline.py`가 단일 타임라인으로 병합한다 (문제 D).

```bash
python examples/demo_full_rehearsal.py            # 시나리오 5종 전체
python examples/demo_full_rehearsal.py --scenario S5   # 총력전 + 통합 타임라인
python scripts/baseline_switch_comparison.py      # 채점기 vs 베이스라인 표
```

| 시나리오 | 검증 내용 | 판정 |
|---|---|---|
| S1 평화 | 전 모듈 on, 오발동 0 | 감지/스위칭/회복 개입 전부 0 |
| S2 충격→우회 | 이중 반응 인터록 (A) | 스위칭 1회, 회복 침묵, 목표 도달 |
| S3 고착→회복 | STALL 예외 경로 | LLM 경로 적용, 오차 절반 이하 |
| S4 마찰 지옥 | 보정 기여도 (3배 마찰) | 추종 오차 +92% 개선 |
| S5 총력전 | 캘리브→충격→고착 연쇄 + (B)(D) | 단일 타임라인 재구성 |

베이스라인 비교(발표 자료: [docs/baseline_comparison.md](docs/baseline_comparison.md)):
채점기 100% / 무작위 86% / 항상-첫-후보 57% 목표 도달 (7시드, 후보 순서 셔플).

통합에서 발견·해결한 상호작용 결함 (개별 데모에선 안 보이던 것):
- **해제 잔향**: 외란이 사라지는 순간도 반대 부호의 충격처럼 보인다. 그 시점엔
  참조가 전진해 접근 후보가 전부 veto → retreat로 임무 포기. 처짐 추세
  (줄어드는 중 = 이미 사라진 외란)로 판별해 무시한다.
- **긴 충격 = 지속 저항**: 1초 넘게 미는 힘은 밀린 채 정지한 관절에서 STALL을
  정당하게 발화시켜 회복 루프까지 개입한다. 충격 시나리오와 저항 시나리오는
  물리적으로 다른 사건이다 (S2 vs S3).

## 확정 하드웨어: 엔젤로보틱스 phact-401 (2026-08 공개)

- 6축, FDCAN, **phorce SDK**(C++/Python/ROS2), Jetson AGX (Ubuntu 22.04, aarch64)
- 액추에이터 내장: **AFC(액티브 마찰제거)**, 하드웨어 동작제한
- 스펙 상수는 [robot_core/hal/profiles.py](robot_core/hal/profiles.py)의 `PHACT_401` 하나만 본다.
  **토크 한계는 용도별 3단** — 원칙: *신호의 지속 시간 특성이 어느 토크 대역을
  쓸 수 있는지 결정한다* (과도 신호는 순간 정격까지, 지속 신호는 연속 예산 안):

  | 한계 | 값 | 용도 |
  |---|---|---|
  | `tau_clamp` | 27×0.8 = **21.6 Nm** | HAL 포화/최종 출력 클램프 (순간 상한 — 버스트는 설계 의도) |
  | `tau_detect` | 7.2×1.5 = **10.8 Nm** | TORQUE_SPIKE 감지 임계 (자체 전이 과도 토크 위 — 대조 테스트로 실증) |
  | `tau_veto` | 27×0.8 = **21.6 Nm** | 스위칭 dream veto 전용 |
  | `tau_cont` | 7.2×0.8 = **5.76 Nm** | 지속 신호 예산: 보정 Δτ 클램프·수집 중단 기준·**1초 이동평균 감시** |
  | `qd_limit` | 15.7×0.8 = **12.56 rad/s** | 속도 한계 |

- **지속 과부하 감시 (CONTINUOUS_OVERLOAD)**: `tau_cont`는 순간 상한이 아니라
  1초 이동평균 예산이다. 신호는 **클램프 적용 후 지령 토크**(발열 = 모터 전류는
  지령 쪽 — 외란과 맞서는 홀드는 출력측 순토크가 0이라 측정토크로는 안 보인다).
  초과 시 회복 루프가 **동작 속도 하향**(`chunk_switch.time_scale`, 위상 적분이라
  변경 순간에도 C1 유지), 예산 복귀 시 `OVERLOAD_CLEARED`가 규칙 직행으로
  속도를 **복원**한다 — 내려가는 길과 올라오는 길이 짝으로 있다.
- 통합 스택 연결: **`StackConfig()` 기본값이 곧 phact-401 6축이다** (3단 한계·게인·
  6축 HOME→TASK 딕셔너리·시나리오 외란 크기 전부 phact 스케일). 우회 축은
  `LATERAL_JOINT`(j0 베이스 요), 딕셔너리는 `detour_left/right`로 좌우 스윙.
  미션 공개 후에는 full_stack.py의 `HOME/TASK/LATERAL_JOINT` 세 상수만 갈아끼운다.
- **AFC 가드**: 캘리브레이션 데이터/모델에 수집 당시 AFC 상태와 **출처**
  (queried=SDK 조회 / declared=사람이 선언)가 기록되고, 보정 노드는 현재 상태와
  불일치하면 활성화를 거부한다 (이중/무효 보정 방지). 현재 상태가 unknown이면
  통과하되 `last_refusal="unverified: ..."` 흔적 + 경고를 남긴다.
  **캘리브레이션은 AFC 상태를 모르면 진행하지 않는다**: HAL이 unknown이면
  `--afc {on,off}`로 선언해야 하고, 무인(`--yes`) 모드에서 선언까지 없으면
  에러로 중단한다 — 조용히 unknown 라벨을 붙이면 나중에 가드가 속는다.

### Jetson AGX (aarch64) 준비 절차

입소 전 노트북이 아니라 **Jetson에서** 아래를 한 번 돌려 검증할 것 (환경이 다르다):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install numpy pyyaml pytest        # aarch64 휠 표준 제공 — 문제 없음
pip install torch --index-url https://download.pytorch.org/whl/cpu   # ↓ 실패 가능
python -m pytest -q                    # 전체 테스트가 Jetson에서 도는지
```

- **torch aarch64 CPU 휠이 없거나 설치가 실패해도 치명적이지 않다**:
  MLP는 '학습'만 torch를 쓰고 런타임 추론은 numpy다. torch가 없으면
  캘리브레이션이 물리 모델 경로만으로 진행되고(MLP 자동 생략, 비교표에 표시),
  MLP 관련 테스트는 자동 skip된다. 물리 모델만으로도 목 리허설 기준
  추종 오차 +92% 개선이 나온다 — MLP는 보너스다.
- pip가 소스 빌드를 시도하며 오래 걸리면 즉시 중단하고 물리 경로로 갈 것.

## 캠프 1일차 체크리스트 (최종판)

전제: **Jetson AGX에** 이 저장소 + venv 준비 완료 (위 aarch64 절차로 검증).

| # | 할 일 | 명령/파일 | 예상 소요 |
|---|---|---|---|
| 0 | 짐 풀기 전 리허설 — 전체 테스트가 지금도 도는지 | `python -m pytest -q` | 3분 |
| 1 | phorce SDK 설치 확인 + 예제 실행 (FDCAN 이름/설정은 주최측 정렬값) | real.py 체크리스트 0번 | 15분 |
| 2 | RealRobotHAL 채우기 — SDK에서 (a) 상태 읽기 (b) 임피던스 지령 (c) AFC 조회 세 함수를 찾아 매핑 | [robot_core/hal/real.py](robot_core/hal/real.py) 의 매핑 (a)(b)(c) 순서대로 | 60–90분 |
| 3 | 모터 1개 댐핑 테스트 → 부호 확인 → AFC on/off 체감 → 홀드 테스트 | real.py 체크리스트 5번 | 20분 |
| 4 | 캘리브레이션 (수집→학습→검증→적용). AFC 조회 매핑 전이면 `--afc {on,off}` 선언 필수 | `python scripts/field_calibration.py --hal real --budget-min 8 --afc off` | 30분 |
| 5 | 청크 딕셔너리 채우기 (미션 공개 후) | README "딕셔너리 채우는 절차" | 30분 |
| 6 | 통합 리허설 — 목으로 시나리오 5종 재확인 | `python examples/demo_full_rehearsal.py` | 5분 |
| 7 | 실물 축소 리허설 — S1(평화)부터, kp 낮게 | full_stack.py의 StackConfig로 게인/임계 조정 | 30분 |
| 8 | LLM 경로 점검 — 키 설정 후 모델 문자열 검증 | `python scripts/check_llm_model.py` | 5분 |

주의사항:
- 실물 1kHz 루프 프로세스는 `gc.disable()`(또는 `gc.freeze()`) 후 돌릴 것 —
  파이썬 GC 일시정지가 p95 지터의 주범이다 (통합 테스트에서 실측).
- 실물에서는 `StackConfig.sync_recovery=False` + `agent.start()` 워커 스레드
  (리허설의 동기 모드는 결정적 테스트용이다).
- 인터넷이 불안하면 LLM 없이 간다 — 회복 루프는 규칙 폴백으로 전부 동작한다.
- **`LLMConfig.model = "claude-opus-5"` 는 아직 API로 검증되지 않았다**
  (개발 환경에 `ANTHROPIC_API_KEY` 없음 → 미확인). 현장에서 8번 단계로 확인하고,
  실패하면 유효한 문자열로 교체할 것. 이 값이 틀려도 LLM 호출만 실패하고
  규칙 폴백이 받아내므로 로봇이 멈추지는 않는다.

## 하지 않기로 한 것 (의도적)

- ROS 2 의존성 — [`adapters/ros2_adapter.py`](robot_core/adapters/ros2_adapter.py) 스켈레톤만 두고 현장에서 래핑한다
- Isaac Sim / MuJoCo 연동 — npz 청크 포맷만 임포트 호환으로 설계해 둠
- 다관절 일반화 동역학 (관성 행렬, 코리올리) — 로직 검증에 불필요
- 물리 정확도 튜닝
- 테스트에서 실제 LLM API 호출 — 전부 주입된 fake client로 검증한다
- 스위칭 판단에 디퓨전/신경망/LLM — 순수 기하+동역학 (ms 단위 반응 경로)
- 델타 모델의 강화학습·전관절 커플링·GPU — 관절별 독립 + CPU로 충분
- 델타 모델 사전 학습 — 모델 파일은 현장 데이터에서만 (커밋 금지)

## 알려진 한계

- `scripts/field_calibration.py`의 목은 **3축**이다 (통합 스택은 6축) — 파이프라인
  단계 검증용이라 축 수가 무관해서 그대로 뒀다. 실기(`--hal real`)에는 영향 없음.

## 테스트

```bash
python -m pytest -q                      # 전체
python -m pytest tests/test_hal_mock.py  # 목 시뮬만
python -m tests.test_logger              # dump_text 출력 눈으로 보기
python -m tests.test_logger --update-snapshot   # 포맷 바꿨을 때만
```

`dump_text` 스냅샷은 `tests/snapshots/dump_text.txt`. 포맷을 의도적으로 바꿨을 때만
갱신하고 diff를 눈으로 확인한 뒤 커밋할 것.
