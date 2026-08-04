# 현장 작업 규칙 (캠프용)

캠프에서 여러 명이 같은 저장소를 만진다. 이 규칙은 "실물에서 A를 고쳤더니
B가 깨졌는데 원인을 못 찾는" 상황을 막기 위한 것이다.

## 커밋

- **작게, 자주.** 특히 현장에서는 **매핑 지점 하나 수정 = 커밋 하나.**
  커밋이 크면 실물 부작용의 원인을 못 찾는다.
- 접두어:
  | 접두어 | 용도 |
  |---|---|
  | `field:` | 실물 대응 수정 (스모크 결과 반영, 파사드 매핑, 임계값 실측치) |
  | `mission:` | 미션 대응 (태그·계획·카탈로그 데이터) |
  | `fix:` / `feat:` / `test:` / `docs:` | 평소대로 |
- 실물 관련 수정은 커밋 메시지에 **어떤 가정이 틀렸는지** 한 줄 포함:
  ```
  field: watch() 콜백이 메인 스레드에서 불림 — A2 가정 틀림
  ```
  (가정 번호는 scripts/field_smoke.py의 A1~A6)

## 푸시

- 푸시 전 반드시 `pytest -q` 통과. 현장에서 시간이 없으면 **최소한**:
  ```bash
  pytest -q tests/test_scenarios_phorce.py tests/test_supervisor.py
  ```
- **CI 실패 상태로 두고 다음 작업 시작하지 말 것.** 빨간 배지 위에 쌓은
  커밋은 전부 의심 대상이 된다.

## 스냅샷 태그

- `pre-mission` = 미션 공개 전 최종 상태 (목 검증 완료, 실물 미접속).
  캠프에서 바뀐 것 전부:
  ```bash
  git diff pre-mission..HEAD --stat     # 발표 자료 "실물 대응 변경사항" 근거
  ```

## 코드 원칙 (요약 — 자세한 건 README)

- 태그는 어느 진입점이든 TagSafetyGuard를 지난다
- 1kHz 콜백은 저장만, 판단·전송은 2Hz 루프
- Supervisor 내부에 새로 손대지 말 것 — 공개 메서드(snapshot/halt/resume/
  request_intent/operator_cleared)로 충분한지 먼저 생각
- legacy/는 삭제 금지 (참고용 스냅샷)
