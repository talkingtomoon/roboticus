# robot_core 프로젝트 지침

phorce 재생 로봇의 "관찰 → 판단 → 선곡" 시스템. 아키텍처와 현장 절차는
README.md, 사람 작업 규칙은 CONTRIBUTING.md 참고.

## 작업 기본 규칙 (모든 작업에 적용)

1. **작업 완료 시 커밋 + 푸시하고, 커밋 해시와 CI 결과를 보고에 포함한다.**
   CI는 GitHub Actions `tests` 워크플로 (pytest + 리허설 시나리오 5종).
2. **테스트가 깨진 채로 작업을 종료하지 않는다.** `pytest -q` 전부 green이
   종료 조건이다.
3. **대규모 삭제·이동이 포함된 작업은 시작 전에 현재 상태를 커밋해둔다.**
4. 커밋 메시지 접두어는 CONTRIBUTING.md를 따른다 (`field:`/`mission:`/...).
   실물 대응 수정은 "어떤 가정이 틀렸는지" 한 줄 포함.

## 코드 불변식 (어기면 안 되는 것)

- 태그 주입은 진입점 무관 `Supervisor.request_intent()` → TagSafetyGuard 경유
- 1kHz 콜백(`FeedbackCache.push`)에는 저장만 — 판단·전송 금지
- 재생 중 play 호출 금지 (BUSY 폭주) — 경계 확인 후에만 선곡
- 안전 명령("멈춰")은 LLM을 기다리지 않는다
- `robot_core/legacy/`는 삭제 금지 (이동만 허용된 참고용 스냅샷)
- 저수준 phorce API 우회 금지, 모션 파일 생성 금지, rclpy 실제 import 금지

## 자주 쓰는 명령

```bash
pytest -q                                      # 전체 (종료 조건)
python examples/demo_full_rehearsal.py         # 리허설 5종
python -m robot_core.ui.server --demo          # 웹 UI 데모 (:8710, /op 포함)
python scripts/field_smoke.py --mock           # 현장 스모크 리허설
```
