"""임피던스 인터페이스 시절의 코드 스냅샷 — 참고용, 실행 대상 아님.

phorce 인터페이스 확정(play(motion_id) 단일 하행, 궤적/토크/게인 전송 불가)으로
아래 전제가 전부 무효가 되면서 이곳으로 이동했다:

- hal_real.py          : 임피던스 지령 HAL stub (실제 하행 경로 없음)
- control/             : 1kHz ControlLoopRunner (지령 루프 자체가 없음)
- chunks/, switching/  : 궤적 스플라인·블렌딩·time_scale (궤적 주입 불가)
- delta/               : 토크 보정 파이프라인 (tau_ff 전송 경로 없음)
- impedance_nodes.py   : Target/Impedance 그래프 노드
- full_stack.py, scenarios_v1.py, supervisor_v1.py : 구 통합 조립

삭제하지 않은 이유: 알고리즘 아이디어(채점 가중치, 히스테리시스 감지,
강건 회귀, C1 블렌딩 수학)는 재사용 가치가 있다. 특히 excitation/collector는
phorce Studio 교시 보조 도구로 부활할 수 있다.

주의: 새 감지기/이벤트 API로 재배선된 robot_core.recovery와는 호환되지
않을 수 있다. 임포트 경로는 맞춰뒀지만 실행은 보장하지 않는다.
"""
