---
title: robot_core
---

# robot_core — phorce 재생 로봇 코어 인프라

"관찰 → 판단 → 선곡" 아키텍처. 실물 없이 `pytest`만으로 검증됩니다.

- **[운용 UI 리뷰 데모 →](demo/)** — 실제 대시보드와 운용자 페이지를
  목 로봇 세션 녹화로 재생합니다 (폰으로 열어도 됩니다)
- [저장소](https://github.com/talkingtomoon/roboticus) · [README](https://github.com/talkingtomoon/roboticus#readme)
- [선곡 베이스라인 비교표](baseline_comparison.md)

데모 페이지는 `python scripts/build_ui_demo.py`로 재생성됩니다 —
UI 코드(`ui/static/*.html`)를 그대로 읽어 네트워크 폴링만 녹화 재생으로
바꾸므로, 보이는 화면이 실물과 같습니다.
