from abc import ABC, abstractmethod

from .allocation import Allocation


class Op(ABC):

    def __init__(self, p_selector):
        self.policy_selector = p_selector

    @abstractmethod
    def __call__(self, alloc: Allocation):
        pass


class AbstractCanonizer(Op):

    def __init__(self, p_selector):
        super().__init__(p_selector)


class AbstractMapper(Op):  # or Decomposer

    def __init__(self, p_selector):
        super().__init__(p_selector)


class AbstractGrouper(Op):

    def __init__(self, p_selector):
        super().__init__(p_selector)


class AbstractRouter(Op):

    def __init__(self, p_selector):
        super().__init__(p_selector)


class AbstractReducer(Op):  # or Assembler

    def __init__(self, p_selector):
        super().__init__(p_selector)


class NOP(Op):  # or Assembler

    def __init__(self, p_selector):
        super().__init__(p_selector)

    def __call__(self, alloc: Allocation):
        pass
