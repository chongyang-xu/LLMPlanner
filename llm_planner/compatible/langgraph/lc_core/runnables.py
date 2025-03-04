from abc import ABC, abstractmethod
from typing import (
    Generic,
    TypeVar,
)

Input = TypeVar("Input", contravariant=True)
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output", covariant=True)

class Runnable(Generic[Input, Output], ABC):
    pass

class RunnableConfig:
    pass