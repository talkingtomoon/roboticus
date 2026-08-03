"""링버퍼 로거.

주의: 이 패키지 이름은 표준 라이브러리 logging과 겹치지만,
파이썬3의 절대 임포트 규칙 때문에 내부에서 `import logging` 하면
표준 라이브러리가 잡힌다. 혼동을 피하려면 항상
`from robot_core.logging import RingLogger` 형태로 쓸 것.
"""

from robot_core.logging.ring_logger import RingLogger

__all__ = ["RingLogger"]
