"""노드 그래프의 최소 단위.

노드 하나 = 제어 파이프라인의 한 단계. ROS 2 노드에 1:1로 대응되도록
인터페이스를 잡아뒀다 (adapters/ros2_adapter.py 참고).

규약:
- update(inputs) -> dict. inputs에는 외부 입력 + 업스트림 노드들의 출력이 합쳐져 온다.
- params는 flat dict[str, float] 을 권장. SafetyGuard가 float만 다루기 때문에
  LLM이 조정할 파라미터는 반드시 float 스칼라로 둘 것.
- update()는 1kHz 루프 안에서 불린다. 무거운 일(파일, 네트워크) 금지.
"""

from __future__ import annotations


class Node:
    """제어 그래프 노드 베이스.

    서브클래스는 update()만 구현하면 된다. 생성자 시그니처
    (name, params=None, enabled=True)는 YAML 로드가 의존하므로 유지할 것.
    """

    def __init__(self, name: str, params: dict | None = None, enabled: bool = True) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("node name must be a non-empty string")
        self.name = name
        self.enabled = bool(enabled)
        self.params: dict = dict(params or {})

    def update(self, inputs: dict) -> dict:
        """한 스텝 실행. 출력 dict는 다운스트림 노드의 inputs에 합쳐진다."""
        raise NotImplementedError(f"{type(self).__name__}.update")

    def __repr__(self) -> str:  # pragma: no cover
        state = "on" if self.enabled else "OFF"
        return f"{type(self).__name__}({self.name!r}, {state}, params={self.params})"
