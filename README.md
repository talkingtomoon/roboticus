# robot_core — 해커톤 코어 인프라 (phorce 판)

[![tests](https://github.com/talkingtomoon/roboticus/actions/workflows/tests.yml/badge.svg)](https://github.com/talkingtomoon/roboticus/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](pyproject.toml)
[![deps](https://img.shields.io/badge/deps-numpy%20%2B%20pyyaml-lightgrey)](pyproject.toml)

**아키텍처: "관찰 → 판단 → 선곡"** — 노래방 선곡기.

실제 로봇 인터페이스가 확정되면서 판이 바뀌었다:
하행은 `play(motion_id)` 하나뿐 (id 1~50, 궤적/토크/게인 전송 경로 없음, 큐 없음),
상행은 1kHz 12축 피드백(position/velocity/current/**dob**/temp + valid/fault).
모션은 행사 전 phorce Studio 교시로 SD카드에 저장 — 코드로 만들 수 없다.

따라서 우리 시스템의 역할 = **피드백을 관찰하다가, 동작이 끝나는 경계마다
"다음에 몇 번 모션을 틀지" 지능적으로 고르는 것.**

```
                        1kHz (저장만)                 ~2Hz (판단·전송)
  phorce 로봇 ──피드백──▶ FeedbackCache ──▶ DecisionLoop(Supervisor)
      ▲                                        │ 관찰: FailureDetector
      │                                        │   PLAYBACK_STALL/IMPACT/
      └────────── play_async(id) ◀─────────────┤   OVERHEAT/AXIS_FAULT
                                               │ 판단: LLM(태그만) | 규칙 폴백
                                               │   TagSafetyGuard(카탈로그 태그)
                                               └ 선곡: MotionSelector
                                                   태그·진입자세·dob순응·진행
```

```
robot_core/
  hal/         phorce.py — PhorceHAL 경계 + MockPhorceHAL(재생 시뮬 + 주입 API)
               mock.py — 12축 관절 동역학 엔진 (목의 물리)
  catalog/     motion_catalog.py — 교시 모션 메타데이터 (JSON, 적재 슬롯 대조)
  switching/   selector.py — 모션 선택기 · baselines.py — 무작위/첫슬롯
  recovery/    detector.py(4종 감지) · llm_agent.py(태그 결정) ·
               safety.py(TagSafetyGuard) · rules.py(실패→태그 표)
  supervisor.py  2Hz 판단 루프 + 상태기계 (1kHz↔2Hz 분리를 구조로 강제)
  logging/     feedback_cache.py — 1kHz 수신과 판단 사이의 유일한 접점
  integration/ scenarios.py(리허설 5종) · timeline.py(단일 시간축)
  adapters/    phorce_ros2.py — rclpy 주석 스켈레톤 (qos_profile_sensor_data 필수)
  legacy/      임피던스 인터페이스 시절 스냅샷 (참고용 — 삭제 아님)
scripts/       annotate_motion.py · validate_catalog.py · field_smoke.py ·
               baseline_selector_comparison.py · check_llm_model.py · legacy/
tests/         162개 테스트, 실물·실제 API 없이 전부 통과
examples/      demo_full_rehearsal.py · legacy/
```

## 설치

```bash
git clone https://github.com/talkingtomoon/roboticus.git
cd roboticus
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q                                  # 162 tests
python examples/demo_full_rehearsal.py               # 리허설 시나리오 5종
```

의존성은 numpy + pyyaml뿐이다 (Jetson AGX aarch64에서 그대로 돈다).
LLM 경로를 쓰려면 `pip install anthropic` + `ANTHROPIC_API_KEY` — **없어도
전부 동작한다** (규칙 폴백).

## 핵심 규칙 (매뉴얼에서 온 제약 → 코드가 강제)

| 매뉴얼 규칙 | 구조로 강제하는 곳 | 테스트 |
|---|---|---|
| 1kHz 콜백에서는 최신 상태 저장만 | `FeedbackCache.push()` (판단 금지) | `test_push_is_storage_only...` |
| 판단·전송은 느린 루프(~2Hz) | `Supervisor.tick()` | `test_decision_tick_stays_fast...` |
| 재생 중 play = BUSY(5) | 경계 확인 후에만 선곡 — BUSY 발생 자체가 버그 | `test_no_busy_storm...` (호출 수 = 재생 수) |
| 12/13 거절 = 사람 개입 | `WAITING_OPERATOR` 상태 — `operator_cleared()` 전 재시도 금지 | `test_operator_rejection_waits...` |
| valid=False 축 신뢰 금지 | 감지기·선택기 모두 valid 마스크 전제 | `test_invalid_axis_excluded...` |
| 경계 감지 1차 = 핸들 완료 | `play_async` 핸들 이벤트, is_busy 폴링은 유실 폴백 | `test_boundary_fallback...` |
| 피드백 신선도 `kStateFreshLimitMs=1500` | 워치독 — 수신 벽시계 정체 시 자동 halt (마지막 프레임으로 계속 판단 금지) | `test_watchdog_halts...` |

## 안전 계열 (2026-08-05 추가)

- **스레드 모드 검증**: 실물 경로(판단 스레드 + LLM 워커 + 1kHz 수신)를
  통합 테스트로 — 결정 전달의 원자적 소비(`_take_pending`)가 여기서 나왔다
- **소프트 정지**: `halt()/resume()` + `HALTED` 상태. 재생 중 모션은 중단
  불가(인터페이스 제약)이지만 새 play는 절대 안 나간다. 관찰은 계속
- **서킷브레이커**: 같은 (실패,축)이 60초 창 안에 3회면 rest 강등, 5회면 halt.
  '계획 전진'을 리셋 신호로 쓰지 않는다 — 재생 완료는 시간 기준이라
  물리적 실패도 전진시키기 때문. `resume()`(사람 개입)만 리셋
- **preflight 게이트**: `start()`가 시작 전 검사 — 카탈로그/적재/계획 태그
  커버리지, E-stop·EtherCAT(`status()`), 피드백 침묵(QoS 증상). 실패 시
  명확한 메시지와 함께 시작 거부
- **의도 게이트**: `request_intent(tag, urgency)` — 사람/음성/미래 파이프라인의
  주입점. **진입점이 어디든 태그는 전부 TagSafetyGuard를 지난다**
  (STT 오인식 "정지"→"저지" 같은 미지 태그 방어)
- **세션 로그**: `FeedbackCache(event_log_path=...)` — 타임라인을 JSONL로
  스트리밍 (메모리 deque 300개는 데모 한 번이면 찬다)

## 실패 감지 (recovery/detector.py)

| 타입 | 신호 | 대응 (규칙 폴백) |
|---|---|---|
| `IMPACT` | dob_a(외란 관측기) 스파이크 — 자체 추정 제거, 하드웨어 DOB가 정본 | `retreat` |
| `PLAYBACK_STALL` | 재생 중 + 전 축 정지 + 전류로 밀고 있음 (물체에 막힘) | `retry` + slow |
| `OVERHEAT` | temp_c 지속 초과. 해제 시 `OVERHEAT_CLEARED` 짝 발행 | `rest` + stop |
| `AXIS_FAULT` | fault 비트 | `rest` + stop |

히스테리시스/최소지속/refractory/lookback 구조는 구 감지기에서 그대로.
**IMPACT lookback(0.7s)은 판단 주기(0.5s)보다 길다** — 폴 사이에 끝난 짧은
충격도 다음 폴의 구간 스캔이 잡는다.

## LLM 회복 (recovery/llm_agent.py) — 실행 경로 분리

LLM은 **의도 태그만** 고른다 (`{"intent_tag", "urgency", "reasoning",
"confidence"}` JSON 강제). 태그는 TagSafetyGuard(= 카탈로그에 실존하는 태그
집합)를 통과해야 하고, motion_id 최종 결정은 항상 선택기가 한다.
불량 응답(파싱 불가/미지 태그/저신뢰/타임아웃 등 7종 테스트)은 전부
규칙 표(실패 타입 → 태그)로 폴백. 비동기 워커·쿨다운·감사 이력 유지.

## 모션 선택기 (switching/selector.py)

채점(낮을수록 좋음): 태그 적합 + 진입 자세 거리 + **dob 순응**((1-cos)/2 —
순응 0 < 무관 0.5 < 대항 1) + 계획 진행. dream veto는 제거(토크 예측 불가
인터페이스) — 대신 하드 필터: **미적재 id 제외**(코드 4 예방), **진입 거리
초과 제외**(MotionAborted 후 임의 자세에서 엉뚱한 재생 방지).

베이스라인 비교 ([docs/baseline_comparison.md](docs/baseline_comparison.md)):
계획 완주율 **선택기 100% / 무작위 14% / 항상-첫슬롯 0%** (7시드).

## 모션 카탈로그 (catalog/)

젯슨 측 JSON: `{id: {name, tags[], start_pose[12], initial_direction[12],
duration_s, notes}}`. `reconcile(robot.motions())`가 적재 슬롯 정본과 대조 —
JSON에만 있으면 경고+제외, 슬롯에만 있으면 "미등록" 경고.

교시 직후 주석은 반자동:
```bash
python scripts/annotate_motion.py --id 3 --name approach --tags approach,slow
# 1회 재생 녹화 → start_pose/initial_direction/duration 자동 추출
```

## 리허설 시나리오 5종 (integration/scenarios.py)

| | 검증 내용 | 판정 |
|---|---|---|
| S1 평화 | 모션 3개 순차 재생 | 오발동 0, BUSY 0, 계획 완주 |
| S2 충격→우회 | dob 스파이크 → retreat 선곡 | 경계에서 side_step 재생 후 복귀 |
| S3 막힘→재시도 | jam → PLAYBACK_STALL → LLM | slow 변주(insert_slow) 재선곡 |
| S4 과열 | temp 주입 → OVERHEAT | 휴지 삽입 → CLEARED → 계획 복귀 |
| S5 총력전 | 충격 + 거절 12 연쇄 | WAITING_OPERATOR 경유 + 단일 타임라인 |

```bash
python examples/demo_full_rehearsal.py            # 전체
python -m robot_core.integration.scenarios --scenario S5   # 타임라인 포함
```

## 캠프 1일차 체크리스트

전제: Jetson AGX에 이 저장소 + venv (`pip install -e ".[dev]"`) + phorce SDK.

| # | 할 일 | 명령/파일 | 예상 소요 |
|---|---|---|---|
| 0 | 짐 풀기 전 — 전체 테스트/리허설이 지금도 도는지 | `pytest -q` + `python examples/demo_full_rehearsal.py` | 5분 |
| 1 | **현장 호환성 스모크** — 파사드 가정 6개 실측 (sim:demo 대상) | `python scripts/field_smoke.py` | 10분 |
| 2 | RealPhorceHAL 작성 — 파사드를 PhorceHAL 인터페이스로 래핑 (스모크가 깨진 가정의 코드 위치를 알려준다) | [robot_core/hal/phorce.py](robot_core/hal/phorce.py) `PhorceHAL` 계약 | 30–60분 |
| 3 | 교시된 모션 주석 달기 (모션당 1회 재생) | `python scripts/annotate_motion.py --id N --name ... --tags ...` | 모션당 2분 |
| 3.5 | **교시 세션 끝나기 전에** 카탈로그·계획 정적 검증 (오타/커버리지/회피 양방향/모션 길이) — 문제는 지금 알아야 다시 가르칠 시간이 있다 | `python scripts/validate_catalog.py --catalog catalog.json --plan approach,insert,...` | 2분 |
| 4 | 감지 임계 반영 — 스모크 A6이 제안한 값으로 | [recovery/detector.py](robot_core/recovery/detector.py) `DetectorConfig` | 5분 |
| 5 | 임무 계획 정의 — 태그 시퀀스 | `MissionPlan([...])` | 5분 |
| 6 | LLM 경로 점검 (키 설정 후 모델 문자열 검증) | `python scripts/check_llm_model.py` | 5분 |
| 7 | 실물 축소 리허설 — 안전한 모션 2개로 S1부터 | `Supervisor` + `sync_recovery=False` + `agent.start()` | 30분 |

주의:
- 실물 운용은 `SupervisorConfig.sync_recovery=False` + `agent.start()` 워커
  (동기 모드는 결정적 리허설/테스트용이다).
- `LLMConfig.model = "claude-opus-5"`는 **API로 검증되지 않았다** (개발 환경에
  키 없음 → 미확인). 6번 단계에서 확인, 실패 시 교체. 틀려도 규칙 폴백으로 돈다.
- ROS 2 직결이 필요해지면 [adapters/phorce_ros2.py](robot_core/adapters/phorce_ros2.py)
  스켈레톤 — **`qos_profile_sensor_data` 안 쓰면 조용히 0개 수신**한다.

## 하지 않기로 한 것 (의도적)

- 저수준 API 우회 — 의도적 비공개라고 명시됨. 시도하지 않는다
- 모션 파일(CSV) 생성 — phorce Studio 전용
- rclpy 실제 import — 주석 스켈레톤만
- 스위칭/선곡 판단에 신경망 — 순수 기하 + 메타데이터. LLM은 태그 선정(초 단위)만
- 테스트에서 실제 LLM API 호출 — 전부 주입된 fake client

## legacy/ — 임피던스 시절 스냅샷

이전 판(임피던스 지령 가정)의 전체 코드는 [robot_core/legacy/](robot_core/legacy/)에
있다 (참고용, 실행 비보장). 채점 가중치·히스테리시스 감지·강건 회귀 같은
알고리즘 아이디어는 현행 코드에 이식됐고, excitation/collector는 교시 보조로
부활할 수 있다. 삭제하지 말 것.

## 알려진 한계

- MockPhorceHAL의 재생 완료는 시간 기준이다 (실물과 동일) — 물리적으로 목표에
  못 갔어도 핸들은 완료된다. "완료 ≠ 성공"의 판별은 감지기(스톨)와 다음 선곡의
  진입 필터가 맡는다.
- 온도 모델은 1차 근사 (전류² 발열 - 선형 냉각) — OVERHEAT 경로 검증용이지
  열 예측용이 아니다. 실물 임계는 스모크 A6 분포를 보고 정한다.
