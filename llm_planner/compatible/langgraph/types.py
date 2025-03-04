import dataclasses
import sys

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Hashable,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

All = Literal["*"]


class BaseCheckpointSaver:
    pass

Checkpointer = Union[None, bool, BaseCheckpointSaver]
"""Type of the checkpointer to use for a subgraph.
- True enables persistent checkpointing for this subgraph.
- False disables checkpointing, even if the parent graph has a checkpointer.
- None inherits checkpointer from the parent graph."""

class Send:
    """A message or packet to send to a specific node in the graph.

    The `Send` class is used within a `StateGraph`'s conditional edges to
    dynamically invoke a node with a custom state at the next step.

    Importantly, the sent state can differ from the core graph's state,
    allowing for flexible and dynamic workflow management.

    One such example is a "map-reduce" workflow where your graph invokes
    the same node multiple times in parallel with different states,
    before aggregating the results back into the main graph's state.

    Attributes:
        node (str): The name of the target node to send the message to.
        arg (Any): The state or message to send to the target node.

    Examples:
        >>> from typing import Annotated
        >>> import operator
        >>> class OverallState(TypedDict):
        ...     subjects: list[str]
        ...     jokes: Annotated[list[str], operator.add]
        ...
        >>> from langgraph.types import Send
        >>> from langgraph.graph import END, START
        >>> def continue_to_jokes(state: OverallState):
        ...     return [Send("generate_joke", {"subject": s}) for s in state['subjects']]
        ...
        >>> from langgraph.graph import StateGraph
        >>> builder = StateGraph(OverallState)
        >>> builder.add_node("generate_joke", lambda state: {"jokes": [f"Joke about {state['subject']}"]})
        >>> builder.add_conditional_edges(START, continue_to_jokes)
        >>> builder.add_edge("generate_joke", END)
        >>> graph = builder.compile()
        >>>
        >>> # Invoking with two subjects results in a generated joke for each
        >>> graph.invoke({"subjects": ["cats", "dogs"]})
        {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
    """

    __slots__ = ("node", "arg")

    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        """
        Initialize a new instance of the Send class.

        Args:
            node (str): The name of the target node to send the message to.
            arg (Any): The state or message to send to the target node.
        """
        self.node = node
        self.arg = arg

    def __hash__(self) -> int:
        return hash((self.node, self.arg))

    def __repr__(self) -> str:
        return f"Send(node={self.node!r}, arg={self.arg!r})"

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Send)
            and self.node == value.node
            and self.arg == value.arg
        )

if sys.version_info >= (3, 10):
    _DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}
else:
    _DC_KWARGS = {"frozen": True}

def default_retry_on(exc: Exception) -> bool:
    import httpx
    import requests

    if isinstance(exc, ConnectionError):
        return True
    if isinstance(
        exc,
        (
            ValueError,
            TypeError,
            ArithmeticError,
            ImportError,
            LookupError,
            NameError,
            SyntaxError,
            RuntimeError,
            ReferenceError,
            StopIteration,
            StopAsyncIteration,
            OSError,
        ),
    ):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        return 500 <= exc.response.status_code < 600 if exc.response else True
    return True


class RetryPolicy(NamedTuple):
    """Configuration for retrying nodes."""

    initial_interval: float = 0.5
    """Amount of time that must elapse before the first retry occurs. In seconds."""
    backoff_factor: float = 2.0
    """Multiplier by which the interval increases after each retry."""
    max_interval: float = 128.0
    """Maximum amount of time that may elapse between retries. In seconds."""
    max_attempts: int = 3
    """Maximum number of attempts to make before giving up, including the first."""
    jitter: bool = True
    """Whether to add random jitter to the interval between retries."""
    retry_on: Union[
        Type[Exception], Sequence[Type[Exception]], Callable[[Exception], bool]
    ] = default_retry_on
    """List of exception classes that should trigger a retry, or a callable that returns True for exceptions that should trigger a retry."""

class ToolOutputMixin:  # type: ignore[no-redef]
    pass

N = TypeVar("N", bound=Hashable)


@dataclasses.dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    """One or more commands to update the graph's state and send messages to nodes.

    Args:
        graph: graph to send the command to. Supported values are:

            - None: the current graph (default)
            - Command.PARENT: closest parent graph
        update: update to apply to the graph's state.
        resume: value to resume execution with. To be used together with [`interrupt()`][langgraph.types.interrupt].
        goto: can be one of the following:

            - name of the node to navigate to next (any node that belongs to the specified `graph`)
            - sequence of node names to navigate to next
            - `Send` object (to execute a node with the input provided)
            - sequence of `Send` objects
    """

    graph: Optional[str] = None
    update: Optional[Any] = None
    resume: Optional[Union[Any, dict[str, Any]]] = None
    goto: Union[Send, Sequence[Union[Send, str]], str] = ()

    def __repr__(self) -> str:
        # get all non-None values
        contents = ", ".join(
            f"{key}={value!r}"
            for key, value in dataclasses.asdict(self).items()
            if value
        )
        return f"Command({contents})"

    def _update_as_tuples(self) -> Sequence[tuple[str, Any]]:
        if isinstance(self.update, dict):
            return list(self.update.items())
        elif isinstance(self.update, (list, tuple)) and all(
            isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str)
            for t in self.update
        ):
            return self.update
        elif hints := get_type_hints(type(self.update)):
            return [(k, getattr(self.update, k)) for k in hints]
        elif self.update is not None:
            return [("__root__", self.update)]
        else:
            return []

    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
