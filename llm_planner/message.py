from typing import Dict, Any, Union


class Message:

    def __init__(self, content: Dict[str, Any] = None, prompt=None) -> None:
        self._content: Dict[str, Any] = content if content is not None else {}
        if prompt is not None:
            self._content["prompt"] = prompt

    # Getter for dictionary-style access
    def __getitem__(self, key: str) -> Any:
        return self._content[key]

    # Setter for dictionary-style access
    def __setitem__(self, key: str, value: Any) -> None:
        self._content[key] = value
