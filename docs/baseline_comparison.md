# 청크 스위칭: 채점기 vs 베이스라인 비교

phact-401 6축, S2형 시나리오 (이동 중 측면 충격 14.5~17.5 Nm, 0.3 s) × 7 시드.
베이스라인은 DREAM-Chunk 원문의 비교 기준 차용: 무작위 선택 / 항상 첫 후보.

| 방식 | 목표 도달률 | 평균 도달 시간 [s] | 최대 토크 평균 [Nm] | 최대 토크 최악 [Nm] |
|---|---|---|---|---|
| **채점기 (본 구현)** | 100% | 2.17 | 16.1 | 17.4 |
| 무작위 선택 | 86% | 2.35 | 16.1 | 17.4 |
| 항상 첫 후보 | 57% | 2.17 | 16.1 | 17.4 |

## 시드별 상세

| 방식 | 시드 | 도달 | 시간 [s] | 최대토크 [Nm] | 선택 청크 |
|---|---|---|---|---|---|
| scorer | 0 | O | 2.26 | 15.4 | detour_left |
| scorer | 1 | O | 2.05 | 17.4 | detour_left |
| scorer | 2 | O | 2.26 | 15.5 | detour_left |
| scorer | 3 | O | 2.19 | 15.3 | detour_left |
| scorer | 4 | O | 2.19 | 16.1 | detour_left |
| scorer | 5 | O | 2.19 | 17.0 | detour_left |
| scorer | 6 | O | 2.05 | 15.6 | detour_left |
| random | 0 | O | 2.26 | 15.4 | detour_left |
| random | 1 | O | 2.05 | 17.4 | direct |
| random | 2 | O | 2.26 | 15.5 | detour_left |
| random | 3 | O | 2.76 | 15.3 | direct_slow |
| random | 4 | X | nan | 16.1 | retreat |
| random | 5 | O | 2.76 | 17.0 | direct_slow |
| random | 6 | O | 2.05 | 15.6 | detour_left |
| first | 0 | O | 2.26 | 15.4 | direct |
| first | 1 | X | nan | 17.4 | retreat |
| first | 2 | X | nan | 15.5 | retreat |
| first | 3 | X | nan | 15.3 | retreat |
| first | 4 | O | 2.19 | 16.1 | detour_right |
| first | 5 | O | 2.19 | 17.0 | detour_right |
| first | 6 | O | 2.05 | 15.6 | direct |
