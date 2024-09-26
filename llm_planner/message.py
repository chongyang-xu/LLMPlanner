from typing import Dict, Any, Union


class Message:
    _id_counter = 0  # Class variable to keep track of the unique id

    def __init__(self,
                 content: Dict[str, Any] = None,
                 prompt=None,
                 uidx=0) -> None:
        self._content: Dict[str, Any] = content if content is not None else {}
        if prompt is not None:
            self._content["prompt"] = prompt
        self.id = 0

        # Increment the class-level id counter and assign it to the instance
        Message._id_counter += 1
        self.id = Message._id_counter

        self.uidx = uidx

    # Getter for dictionary-style access
    def __getitem__(self, key: str) -> Any:
        if key in self._content:
            return self._content[key]
        else:
            return None

    # Setter for dictionary-style access
    def __setitem__(self, key: str, value: Any) -> None:
        self._content[key] = value

    def __repr__(self):
        return str(self._content)
