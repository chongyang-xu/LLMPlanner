import sys

from llm_planner.compatible.langgraph.types import Send

EMPTY_SEQ: tuple[str, ...] = tuple()
"""Tag to hide a node/edge from certain tracing/streaming environments."""
START = sys.intern("__start__")
"""The first (maybe virtual) node in graph-style Pregel."""
END = sys.intern("__end__")

TAG_HIDDEN = sys.intern("langsmith:hidden")
