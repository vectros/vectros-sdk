"""The Agent: an LLM loop that runs on the AIOS kernel.

    agent = Agent("helper", tools=[weather])
    print(agent.run("weather in Pune?"))

Every model call is an AIOS LLM syscall, so the kernel schedules, meters and
traces it. Kernel-registered tools (named by string) run through the AIOS
tool worker with kernel approvals. ``@tool`` functions run in this process.
"""

from __future__ import annotations

import json
import os
import re
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ._tracing import TraceSpan, flush_spans
from .errors import KernelError, StepLimitReached, ToolDenied, VectrosError
from .tools import KernelTool, Tool

Approver = Callable[[str, dict], bool]

OBSERVATION_MAX = 8000
HISTORY_MAX = 40

_PROTOCOL = """\
You can use tools. Reply with exactly one JSON object and nothing else:
{"tool": "<tool name>", "args": {<arguments>}} to call a tool, or
{"answer": "<your final answer for the user>"} when you are done.

Tools:
"""


@dataclass
class Event:
    """One step of a streamed run.

    kind is "token" (model output text), "tool_call", "tool_result" or
    "answer". data holds the text, or {"tool", "args"} / {"tool", "result"}.
    """

    kind: str
    data: Any


def _session_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "vectros" / "sessions"


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value) or value.startswith("."):
        raise ValueError(f"invalid name {value!r}: use 1-64 of A-Z a-z 0-9 . _ -")
    return value


def _extract_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _function_spec(item: Tool | KernelTool) -> dict:
    return {"type": "function", "function": {
        "name": item.name, "description": item.description, "parameters": item.parameters}}


def _parse_envelope(text: str, step: int) -> tuple[str, list[dict]]:
    """Split the LLM worker's tool envelope into content and tool calls.
    A plain-text reply (older worker) is a final answer."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text, []
    if not isinstance(value, dict) or value.get("aios_llm_result") != 1:
        return text, []
    calls = []
    for index, call in enumerate(value.get("tool_calls") or []):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        calls.append({"id": call.get("id") or f"call_{step}_{index}", "type": "function",
                      "function": {"name": str(function["name"]),
                                   "arguments": function.get("arguments") or "{}"}})
    return str(value.get("content") or "").strip(), calls


# Some models (gemma4 among them) now and then return no text after reading
# tool results; the worker reports that as EIO "no content".
_FINAL_NUDGE = "Using the tool results above, give your final answer to the user now."


def _empty_reply(exc: KernelError) -> bool:
    return exc.errno == 5 and "no content" in str(exc)


def _parse_arguments(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return None


def ask_terminal(tool_name: str, args: dict) -> bool:
    """Approver that asks on the terminal: ``Agent(..., approve=ask_terminal)``."""
    answer = input(f"Allow tool {tool_name} {json.dumps(args, ensure_ascii=False)}? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


class Storage:
    """The agent's private AIOS storage (versioned, quota-limited by the kernel)."""

    def __init__(self, agent: "Agent"):
        self._agent = agent

    def write(self, path: str, data: str | bytes) -> None:
        kernel, agent_id = self._agent._connect(storage=True)
        kernel.storage_write(agent_id, path, data.encode() if isinstance(data, str) else data)

    def read(self, path: str) -> bytes:
        kernel, agent_id = self._agent._connect(storage=True)
        return kernel.storage_read(agent_id, path)


class Agent:
    """An AI agent registered with the AIOS kernel.

    Args:
        name: Agent name shown in AIOS Manager and Trace.
        model: Model to request; it must be on the kernel's model allowlist.
            Default: the LLM worker's configured model.
        tools: ``@tool`` functions and/or names of kernel-registered tools.
        system: System prompt.
        session: Keep the conversation under this name across runs and restarts.
        max_steps: Maximum model calls per run.
        approve: Called as ``approve(tool_name, args) -> bool`` for tools that
            need approval. Without it, the call waits for the owner to decide
            in AIOS Manager.
        timeout: Seconds allowed for each model or tool call.
        tool_protocol: "native" uses the model's function calling; "json"
            asks for JSON replies instead, for models without tool support.
    """

    def __init__(self, name: str, *, model: str | None = None,
                 tools: Iterable[Tool | str] = (), system: str | None = None,
                 session: str | None = None, max_steps: int = 10,
                 approve: Approver | None = None, timeout: float = 120,
                 tool_protocol: str = "native", kernel: Any = None):
        if tool_protocol not in ("native", "json"):
            raise ValueError('tool_protocol must be "native" or "json"')
        self.tool_protocol = tool_protocol
        self.name = _safe_name(name)
        self.model = model
        self.system = system
        self.session = _safe_name(session) if session else None
        self.max_steps = max_steps
        self.approve = approve
        self.timeout_ms = int(timeout * 1000)
        self.storage = Storage(self)
        self._tool_specs = list(tools)
        self._tools: dict[str, Tool | KernelTool] | None = None
        self._kernel = kernel
        self._owns_kernel = kernel is None
        self._agent_id: int | None = None
        self._storage_ready = False
        self.history: list[dict] = self._load_session()

    # --- lifecycle ---

    def _connect(self, storage: bool = False):
        if self._kernel is None:
            from ._kernel import Kernel
            self._kernel = Kernel()
        if self._agent_id is None:
            from ._kernel import SYSCALL_LLM, SYSCALL_TOOL
            self._agent_id = self._kernel.register(self.name)
            self._finalizer = weakref.finalize(self, _release, self._kernel, self._agent_id,
                                               self._owns_kernel)
            self._kernel.setup_core(self._agent_id, SYSCALL_LLM, self.model or "")
            if self._tool_specs:
                self._kernel.setup_core(self._agent_id, SYSCALL_TOOL)
        if storage and not self._storage_ready:
            from ._kernel import SYSCALL_STORAGE
            self._kernel.setup_core(self._agent_id, SYSCALL_STORAGE)
            self._storage_ready = True
        return self._kernel, self._agent_id

    def close(self) -> None:
        """Unregister from the kernel. Called automatically at exit."""
        if self._agent_id is not None:
            self._finalizer()
            self._agent_id = None
            if self._owns_kernel:
                self._kernel = None
            self._storage_ready = False

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def id(self) -> int:
        """Kernel agent ID (registers the agent if needed)."""
        return self._connect()[1]

    # --- tools ---

    def _resolve_tools(self) -> dict[str, Tool | KernelTool]:
        if self._tools is not None:
            return self._tools
        resolved: dict[str, Tool | KernelTool] = {}
        names = [spec for spec in self._tool_specs if isinstance(spec, str)]
        registry = {}
        if names:
            kernel, _ = self._connect()
            registry = {entry["name"]: entry for entry in kernel.list_tools()}
        for spec in self._tool_specs:
            if isinstance(spec, Tool):
                resolved[spec.name] = spec
            elif isinstance(spec, str):
                if spec not in registry:
                    known = ", ".join(sorted(registry)) or "none"
                    raise VectrosError(f"kernel tool {spec!r} is not registered (available: {known})")
                entry = registry[spec]
                schema = dict(entry["schema"])
                description = schema.pop("description", "") or spec
                schema.setdefault("type", "object")
                schema.setdefault("properties", {})
                resolved[spec] = KernelTool(spec, description, schema, entry["approval_required"])
            elif callable(spec):
                raise TypeError(f"{getattr(spec, '__name__', spec)!r} is not a tool; decorate it with @tool")
            else:
                raise TypeError(f"unsupported tool {spec!r}")
        self._tools = resolved
        return resolved

    def _system_prompt(self, tools: dict[str, Tool | KernelTool]) -> str | None:
        if not tools:
            return self.system
        lines = []
        for item in tools.values():
            note = " (needs approval)" if item.approval else ""
            lines.append(f"- {item.name}: {item.description}{note}\n"
                         f"  parameters: {json.dumps(item.parameters, ensure_ascii=False)}")
        prompt = _PROTOCOL + "\n".join(lines)
        return f"{self.system}\n\n{prompt}" if self.system else prompt

    def _call_tool(self, item: Tool | KernelTool, args: dict, root: TraceSpan | None) -> str:
        kernel, agent_id = self._connect()
        if isinstance(item, KernelTool):
            span = root.child("aios.action", "AGENT") if root else None
            return kernel.tool(agent_id, item.name, args, self.timeout_ms,
                               self.approve, span)
        if getattr(kernel, "supports_client_tools", False):
            span = root.child(f"tool.{item.name}", "TOOL") if root else None
            return kernel.client_tool(agent_id, item.name, args, lambda: item.invoke(args),
                                      item.approval, self.approve, self.timeout_ms, span)
        # Kernels before ABI 4.3 cannot mediate local tools: approve here.
        if item.approval and not (self.approve and self.approve(item.name, args)):
            raise ToolDenied(item.name, "approval required" if not self.approve else "denied by approver")
        span = root.child(f"tool.{item.name}", "TOOL") if root else None
        if span:
            span.set_input({"tool.name": item.name, "tool.arguments": args})
        try:
            result = item.invoke(args)
        except Exception as exc:
            if span:
                span.finish(error=f"{type(exc).__name__}: {exc}")
            raise
        if span:
            span.finish(output=result)
        return result

    # --- running ---

    @staticmethod
    def _relay(reply, emit: bool):
        """Forward streamed text as token events; return the full result."""
        while True:
            try:
                chunk = next(reply)
            except StopIteration as done:
                return (done.value or "").strip()
            if emit:
                yield Event("token", chunk)

    def _run_tool_call(self, tools, name: str, args: Any, root):
        yield Event("tool_call", {"tool": name, "args": args})
        if name not in tools:
            observation = f"error: unknown tool {name!r}; use one of {', '.join(tools)}"
        elif not isinstance(args, dict):
            observation = "error: arguments must be a JSON object"
        else:
            try:
                observation = self._call_tool(tools[name], args, root)
            except ToolDenied as exc:
                observation = f"denied: {exc.reason or exc}"
            except Exception as exc:  # Report the failure back to the model.
                observation = f"error: {type(exc).__name__}: {exc}"
        yield Event("tool_result", {"tool": name, "result": observation})
        return observation[:OBSERVATION_MAX]

    def stream(self, prompt: str) -> Iterator[Event]:
        """Run the agent, yielding tokens, tool calls and the final answer."""
        tools = self._resolve_tools()
        kernel, agent_id = self._connect()
        root = None
        if kernel.trace_enabled(agent_id):
            root = TraceSpan.root(f"agent.{self.name}", "AGENT", agent_id, self.session)
            root.set_input(prompt, "text/plain")
        native = bool(tools) and self.tool_protocol == "native"
        json_protocol = bool(tools) and not native
        system = self._system_prompt(tools) if json_protocol else self.system
        specs = [_function_spec(item) for item in tools.values()] if native else None
        turn: list[dict] = [{"role": "user", "content": prompt}]
        answer = None
        nudged = False
        try:
            for step in range(self.max_steps):
                messages = ([{"role": "system", "content": system}] if system else []) \
                    + self.history + turn
                span = root.child("aios.planning", "AGENT") if root else None
                reply = kernel.llm_stream(agent_id, messages, self.model, json_mode=json_protocol,
                                          timeout_ms=self.timeout_ms, span=span, tools=specs)
                # JSON-protocol steps are machine text; everything else streams.
                try:
                    text = yield from self._relay(reply, emit=not json_protocol)
                except KernelError as exc:
                    # After tool results, ask once more for the answer.
                    if nudged or len(turn) == 1 or not _empty_reply(exc):
                        raise
                    nudged = True
                    turn.append({"role": "user", "content": _FINAL_NUDGE})
                    continue
                if not tools:
                    answer = text
                    break
                if native:
                    content, calls = _parse_envelope(text, step)
                    if not calls:
                        answer = content
                        break
                    turn.append({"role": "assistant", "content": content, "tool_calls": calls})
                    for call in calls:
                        observation = yield from self._run_tool_call(
                            tools, call["function"]["name"],
                            _parse_arguments(call["function"]["arguments"]), root)
                        turn.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": observation})
                    continue
                plan = _extract_json(text)
                if plan is None or "tool" not in plan:
                    answer = str(plan.get("answer", text)) if plan else text
                    break
                name = str(plan["tool"])
                observation = yield from self._run_tool_call(tools, name, plan.get("args") or {}, root)
                turn += [{"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
                         {"role": "user", "content": f"Tool {name} returned:\n{observation}"}]
            if answer is None:
                raise StepLimitReached(f"no answer after {self.max_steps} steps")
        except BaseException as exc:
            if root:
                root.finish(error=f"{type(exc).__name__}: {exc}")
            raise
        if root:
            root.finish(output=answer)
        # Only the user prompt and final answer enter the session history.
        self.history += [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}]
        self.history = self.history[-HISTORY_MAX:]
        self._save_session()
        yield Event("answer", answer)

    def run(self, prompt: str) -> str:
        """Run the agent to completion and return its final answer."""
        answer = ""
        for event in self.stream(prompt):
            if event.kind == "answer":
                answer = event.data
        return answer

    def __call__(self, prompt: str) -> str:
        return self.run(prompt)

    # --- sessions ---

    def _session_path(self) -> Path | None:
        if not self.session:
            return None
        return _session_dir() / self.name / f"{self.session}.json"

    def _load_session(self) -> list[dict]:
        path = self._session_path()
        if path is None or not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _save_session(self) -> None:
        path = self._session_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.history, ensure_ascii=False))
        os.chmod(tmp, 0o600)
        tmp.replace(path)

    def reset(self) -> None:
        """Forget the conversation, including the saved session."""
        self.history = []
        path = self._session_path()
        if path is not None:
            path.unlink(missing_ok=True)


def _release(kernel, agent_id: int, owns_kernel: bool) -> None:
    flush_spans()
    try:
        kernel.unregister(agent_id)
    except Exception:
        pass
    if owns_kernel:
        kernel.close()
