"""LLM 회복 경로 점검 — 모델 문자열이 실제로 존재하는지 확인.

캠프 첫날 체크리스트 8번. 키가 없으면 '미확인'으로 보고하고 넘어간다
(LLM 없이도 회복 루프는 규칙 폴백으로 동작한다).

    python scripts/check_llm_model.py
    python scripts/check_llm_model.py --model claude-sonnet-4-6
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_core.recovery.llm_agent import LLMConfig, check_model  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="검증할 모델 문자열 (기본: LLMConfig)")
    args = p.parse_args()
    cfg = LLMConfig(model=args.model) if args.model else LLMConfig()
    sys.exit(check_model(cfg))
