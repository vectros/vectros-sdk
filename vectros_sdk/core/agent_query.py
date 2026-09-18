"""
SDK.8 -- LangChain-style chainable composition ("|" / ".then()") over the
SDK's typed Query API, producing an **AgentQuery**: an ordered, named
sequence of steps -- the SDK-facing composite unit of work `sdk.png` draws
(name mirrored, in vocabulary only, from `aios_core::kernel::query::AgentQuery`
-- see "Real gap" below for exactly what is and is not shared with that
Rust type).

Per ARCH.23, decomposing one query into a dependency graph (an
`OperationGraph`) is the kernel's `QueryPlanner`'s job, not the SDK's. This
module never builds one: composition here is strictly linear -- each step
is dispatched only after the one before it finishes, threading that step's
real response forward. Structurally this is function composition
(`f(g(x))`), not a scheduler: there is no branching, no parallel edges, and
no notion of "ready when its dependencies are satisfied."

Real gap, found while building this, not routed around: the wire protocol
(`aios_sdk::execution::KernelRequest`) has no Composite/AgentQuery
submission variant at all -- every `ExecutionRequest` carries exactly one
complete effect. So there is no live endpoint an AgentQuery could be
handed to as a single unit even if this module built one, and the kernel's
real `QueryPlanner`/`OperationGraph` (`aios-core/src/kernel/query.rs`) is
unreachable from here. `AgentQuery.run()` is honest about what it actually
does: it dispatches each already-existing, already-decomposed typed Query
(LLMQuery/MemoryQuery/StorageQuery/ToolQuery) one at a time through the
exact same `AIOSClient` sub-clients -- and therefore the same
`send_request(Query)` dispatcher -- every capability module already uses
and already has real tests for, never a client-side re-implementation of
kernel decomposition. Extending the wire protocol with a real Composite
variant, for the kernel's own QueryPlanner to plan, is separate, unscoped
kernel-side work.

Example
-------
>>> chain = (
...     llm_step("ask", [{"role": "user", "content": "Name one AIOS invariant."}],
...               llms=[{"name": "gemma4:e4b"}])
...     | memory_create_step("remember", lambda history: history[-1].response_message)
... )
>>> result = chain.run(client)  # a real AIOSClient(socket_path=...)
>>> result.last.success
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from vectros_sdk.client.client import AIOSClient

_HistoryFn = Callable[[List[Any]], Any]


def _resolve(value: Union[Any, _HistoryFn], history: List[Any]) -> Any:
    """A step argument is either a literal value or a callable taking the
    ordered list of prior steps' real responses -- the mechanism by which
    one step's real output threads into the next step's input."""
    return value(history) if callable(value) else value


class AgentQueryError(Exception):
    """Raised for structural misuse of an AgentQuery pipeline (no steps, no
    socket_path). Never wraps a real dispatch failure -- those propagate
    with their own original type (RealBackendUnsupported, AIOSKernelError,
    MemoryFeatureUnimplemented, ...) so callers can still catch them
    precisely, exactly as they would calling that capability directly."""


class QueryStep:
    """One real, atomic unit of an AgentQuery pipeline: a name (for
    diagnostics) and the AIOSClient call it dispatches."""

    def __init__(self, name: str, call: Callable[["AIOSClient", List[Any]], Any]) -> None:
        self.name = name
        self._call = call

    def __call__(self, client: "AIOSClient", history: List[Any]) -> Any:
        return self._call(client, history)

    def __repr__(self) -> str:
        return f"QueryStep({self.name!r})"

    def then(self, other: "StepOrChain") -> "AgentQuery":
        return AgentQuery([self]).then(other)

    def __or__(self, other: "StepOrChain") -> "AgentQuery":
        return self.then(other)

    def run(self, client: "AIOSClient") -> "AgentQueryResult":
        return AgentQuery([self]).run(client)


class AgentQueryResult:
    """The real responses collected from running an AgentQuery, in step order."""

    def __init__(self, step_names: List[str], responses: List[Any]) -> None:
        self.step_names = step_names
        self.responses = responses

    @property
    def last(self) -> Any:
        if not self.responses:
            raise AgentQueryError("AgentQuery pipeline produced no responses")
        return self.responses[-1]

    def __getitem__(self, index: int) -> Any:
        return self.responses[index]

    def __len__(self) -> int:
        return len(self.responses)

    def __repr__(self) -> str:
        return f"AgentQueryResult(step_names={self.step_names!r})"


StepOrChain = Union[QueryStep, "AgentQuery"]


class AgentQuery:
    """An ordered, named sequence of `QueryStep`s. Build with `.then()`/`|`;
    run for real with `.run(client)`. See the module docstring for exactly
    what "real" means here and the one protocol gap that bounds it."""

    def __init__(self, steps: Optional[List[QueryStep]] = None) -> None:
        self._steps: List[QueryStep] = list(steps or [])

    def then(self, other: StepOrChain) -> "AgentQuery":
        if isinstance(other, AgentQuery):
            return AgentQuery([*self._steps, *other._steps])
        return AgentQuery([*self._steps, other])

    def __or__(self, other: StepOrChain) -> "AgentQuery":
        return self.then(other)

    @property
    def steps(self) -> List[QueryStep]:
        return list(self._steps)

    def run(self, client: "AIOSClient") -> AgentQueryResult:
        if not self._steps:
            raise AgentQueryError("cannot run an AgentQuery with no steps")
        if not client.socket_path:
            raise AgentQueryError(
                "AgentQuery.run requires an AIOSClient(socket_path=...) -- "
                "composite execution has no HTTP-mock equivalent"
            )
        responses: List[Any] = []
        for step in self._steps:
            responses.append(step(client, responses))
        return AgentQueryResult(
            step_names=[step.name for step in self._steps], responses=responses
        )


# --- Step factories for the four query types named in sdk.png/SDK.8 -------
# Each wraps the real AIOSClient sub-client call already proven against a
# live kernel (see test_real_kernel_client.py) -- this module never
# reconstructs a Query object itself, so there is exactly one place in the
# codebase that knows how to build a valid LLMQuery/MemoryQuery/
# StorageQuery/ToolQuery.


def llm_step(
    name: str,
    messages: Union[List[Dict[str, Any]], _HistoryFn],
    *,
    llms: Optional[List[Dict[str, Any]]] = None,
) -> QueryStep:
    def call(client: "AIOSClient", history: List[Any]) -> Any:
        return client.llm.chat(_resolve(messages, history), llms=llms)

    return QueryStep(name, call)


def memory_create_step(
    name: str,
    content: Union[str, _HistoryFn],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> QueryStep:
    def call(client: "AIOSClient", history: List[Any]) -> Any:
        return client.memory.create(_resolve(content, history), metadata=metadata)

    return QueryStep(name, call)


def storage_write_step(
    name: str,
    file_path: Union[str, _HistoryFn],
    content: Union[str, _HistoryFn],
) -> QueryStep:
    def call(client: "AIOSClient", history: List[Any]) -> Any:
        return client.storage.write_file(
            _resolve(file_path, history), _resolve(content, history)
        )

    return QueryStep(name, call)


def tool_call_step(
    name: str,
    tool_calls: Union[List[Dict[str, Any]], _HistoryFn],
) -> QueryStep:
    def call(client: "AIOSClient", history: List[Any]) -> Any:
        return client.tool.call(_resolve(tool_calls, history))

    return QueryStep(name, call)
