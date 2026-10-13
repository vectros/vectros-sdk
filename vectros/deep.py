"""DeepAgent: an agent that plans, keeps working files and delegates.

    researcher = SubAgent("researcher", "Researches one question in depth.",
                          system="You research thoroughly and cite sources.",
                          tools=["search_web"])
    agent = DeepAgent("analyst", tools=[search], subagents=[researcher])
    print(agent.run("Compare the three largest Linux schedulers"))

On top of the Agent loop, a DeepAgent has built-in tools:

- ``write_todos``: keep a plan and update it as steps finish.
- ``ls``, ``read_file``, ``write_file``, ``edit_file``: a private file space for
  notes and drafts, shared with its subagents. It is kept with the session.
- ``task``: hand a self-contained job to a subagent, which runs with a clean
  context as its own kernel agent and returns only its final answer.

Every built-in tool call is a kernel tool call like any ``@tool`` function.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .agent import Agent, Event, _safe_name
from .config import agent_settings
from .errors import VectrosError
from .tools import Tool, tool

GENERAL_PURPOSE = "general-purpose"
FILE_MAX = 512 * 1024
READ_LIMIT = 2000

BASE_PROMPT = """\
You are an agent that completes complex, multi-step tasks.

Planning: for any task with more than two steps, call write_todos first with
the steps, mark one step in_progress while you work on it and completed as
soon as it is done. Update the list when the plan changes.

Files: use write_file, read_file, edit_file and ls to keep notes, collected
facts and drafts instead of holding everything in the conversation. Read a
file before you edit it.

Subagents: use task to hand a self-contained piece of work to a subagent. Give
it every detail it needs; it cannot see this conversation. It returns only its
final answer. Hand off independent pieces one at a time and combine the
results yourself.

When the work is done, give the user a complete final answer."""


@dataclass
class SubAgent:
    """A specialist the DeepAgent can hand work to with the ``task`` tool.

    Args:
        name: Name used in ``task(subagent=...)``; the kernel agent is
            ``<parent>.<name>``.
        description: When to use this subagent (shown to the parent model).
        system: The subagent's system prompt.
        tools: ``@tool`` functions, kernel tool names or MCP servers.
        model: Model for the subagent. Default: the parent's model.
    """

    name: str
    description: str
    system: str = ""
    tools: Iterable[Any] = field(default_factory=tuple)
    model: str | None = None


def _number_lines(text: str, offset: int, limit: int) -> str:
    lines = text.splitlines()
    chosen = lines[offset:offset + limit]
    out = "\n".join(f"{offset + index + 1:6}\t{line[:2000]}" for index, line in enumerate(chosen))
    if offset + limit < len(lines):
        out += f"\n... {len(lines) - offset - limit} more lines; read with offset={offset + limit}"
    return out


class DeepAgent(Agent):
    """An Agent with planning, a file space and subagents.

    Takes every Agent argument, plus:

    Args:
        subagents: SubAgent specialists for the ``task`` tool. A
            "general-purpose" subagent with this agent's tools is always there.
        max_steps: Maximum model calls per run (default 50).
    """

    DEFAULTS = {**Agent.DEFAULTS, "max_steps": 50}
    BUILTIN = ("write_todos", "ls", "read_file", "write_file", "edit_file", "task")

    def __init__(self, name: str, *, tools: Iterable[Any] = (), system: str | None = None,
                 subagents: Iterable[SubAgent] = (),
                 on_subagent_event: Callable[[str, Event], None] | None = None, **kwargs):
        self.todos: list[dict] = []
        # Called as on_subagent_event(subagent, event) for each step a
        # subagent takes, so a caller can show work the task tool hides.
        self.on_subagent_event = on_subagent_event
        self.files: dict[str, str] = {}
        self._user_tools = list(tools)
        self._subagent_specs: dict[str, SubAgent] = {}
        for spec in subagents:
            _safe_name(spec.name)
            if spec.name in self._subagent_specs or spec.name == GENERAL_PURPOSE:
                raise ValueError(f"duplicate subagent name {spec.name!r}")
            self._subagent_specs[spec.name] = spec
        self._subagents: dict[str, Agent] = {}
        if system is None:  # The prompt is built here, so read the configured one first.
            system = agent_settings(_safe_name(name)).get("system")
        self._user_system = system or ""
        prompt = BASE_PROMPT + self._subagent_list()
        if system:
            prompt = f"{system}\n\n{prompt}"
        super().__init__(name, tools=[*self._user_tools, *self._builtin_tools()],
                         system=prompt, **kwargs)
        self._load_state()

    # --- built-in tools ---

    def _subagent_list(self) -> str:
        lines = [f"- {GENERAL_PURPOSE}: any self-contained task; has the same tools as you."]
        lines += [f"- {spec.name}: {spec.description}" for spec in self._subagent_specs.values()]
        return "\n\nSubagents for task:\n" + "\n".join(lines)

    def _file_tools(self) -> list[Tool]:
        files = self.files

        @tool
        def ls() -> str:
            """List the files in the agent's file space with their sizes."""
            if not files:
                return "no files"
            return "\n".join(f"{path}\t{len(text)} chars" for path, text in sorted(files.items()))

        @tool
        def read_file(path: str, offset: int = 0, limit: int = READ_LIMIT) -> str:
            """Read a file, with line numbers.

            Args:
                path: File path, e.g. "notes/plan.md".
                offset: First line to read (0-based).
                limit: Maximum number of lines.
            """
            if path not in files:
                raise FileNotFoundError(f"no file {path!r}; use ls")
            text = files[path]
            return _number_lines(text, max(0, offset), max(1, limit)) if text else "(empty file)"

        @tool
        def write_file(path: str, content: str) -> str:
            """Create or replace a file.

            Args:
                path: File path, e.g. "notes/plan.md".
                content: The whole file content.
            """
            if not path or len(path) > 255 or path.startswith("/") or "\0" in path:
                raise ValueError("path must be 1-255 characters, relative, without NUL")
            if len(content) > FILE_MAX:
                raise ValueError(f"file larger than {FILE_MAX} characters")
            files[path] = content
            return f"wrote {path} ({len(content)} chars)"

        @tool
        def edit_file(path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> str:
            """Replace exact text in a file. old_string must occur exactly once
            unless replace_all is true.

            Args:
                path: File to edit.
                old_string: Exact text to replace.
                new_string: Replacement text.
                replace_all: Replace every occurrence.
            """
            if path not in files:
                raise FileNotFoundError(f"no file {path!r}; use ls")
            count = files[path].count(old_string) if old_string else 0
            if count == 0:
                raise ValueError("old_string not found in the file")
            if count > 1 and not replace_all:
                raise ValueError(f"old_string occurs {count} times; add context or set replace_all")
            text = files[path].replace(old_string, new_string, -1 if replace_all else 1)
            if len(text) > FILE_MAX:
                raise ValueError(f"file larger than {FILE_MAX} characters")
            files[path] = text
            return f"edited {path} ({count if replace_all else 1} replacement(s))"

        return [ls, read_file, write_file, edit_file]

    def _builtin_tools(self) -> list[Tool]:
        @tool
        def write_todos(todos: list[dict]) -> str:
            """Replace the plan. Each item is {"content": str, "status":
            "pending" | "in_progress" | "completed"}.

            Args:
                todos: The full, updated list of steps.
            """
            plan = []
            for item in todos:
                if not isinstance(item, dict) or not str(item.get("content", "")).strip():
                    raise ValueError('each todo needs a "content" string')
                status = item.get("status", "pending")
                if status not in ("pending", "in_progress", "completed"):
                    raise ValueError(f"invalid status {status!r}")
                plan.append({"content": str(item["content"]).strip(), "status": status})
            self.todos = plan
            marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
            return "\n".join(f"{marks[t['status']]} {t['content']}" for t in plan) or "plan cleared"

        @tool
        def task(description: str, subagent: str = GENERAL_PURPOSE) -> str:
            """Hand a self-contained task to a subagent and get its final answer.

            Args:
                description: The full task, with every detail the subagent needs.
                subagent: Which subagent to use.
            """
            return self._delegate(subagent, description)

        return [write_todos, *self._file_tools(), task]

    def _resolve_tools(self):
        if self._tools is None:
            for spec in self._user_tools:
                if isinstance(spec, (Tool, str)) and getattr(spec, "name", spec) in self.BUILTIN:
                    raise VectrosError(f"tool name {getattr(spec, 'name', spec)!r} is "
                                       "reserved by DeepAgent")
        return super()._resolve_tools()

    # --- subagents ---

    def _subagent(self, name: str) -> Agent:
        if name in self._subagents:
            return self._subagents[name]
        if name != GENERAL_PURPOSE and name not in self._subagent_specs:
            known = ", ".join([GENERAL_PURPOSE, *self._subagent_specs])
            raise ValueError(f"unknown subagent {name!r}; use one of {known}")
        # The kernel allows one agent per /dev/aios handle, so a subagent opens
        # its own unless the caller supplied the kernel object.
        kernel = None if self._owns_kernel else self._connect()[0]
        if name == GENERAL_PURPOSE:
            spec = SubAgent(name, "", system=self._user_system, tools=self._user_tools)
        else:
            spec = self._subagent_specs[name]
        # The subagent shares the file space; it does not plan or delegate.
        agent = Agent(f"{self.name}.{name}"[:64], model=spec.model or self.model,
                      tools=[*spec.tools, *self._file_tools()],
                      system=(spec.system + "\n\n" if spec.system else "")
                      + "Complete the task you are given and reply with the result.",
                      max_steps=self.max_steps, approve=self.approve,
                      timeout=self.timeout_ms / 1000, tool_protocol=self.tool_protocol,
                      trace=self.trace, kernel=kernel)
        self._subagents[name] = agent
        return agent

    def _delegate(self, name: str, description: str) -> str:
        agent = self._subagent(name)
        agent.history = []  # Every task starts from a clean context.
        if self.on_subagent_event is None:
            return agent.run(description)
        answer = ""
        for event in agent.stream(description):
            if event.kind == "answer":
                answer = event.data
            self.on_subagent_event(name, event)
        return answer

    def close(self) -> None:
        for agent in self._subagents.values():
            agent.close()
        self._subagents.clear()
        super().close()

    # --- sessions ---

    def _state_path(self):
        path = self._session_path()
        return path.with_suffix(".deep.json") if path else None

    def _load_state(self) -> None:
        path = self._state_path()
        if path is None or not path.exists():
            return
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(state, dict):
            self.todos = list(state.get("todos") or [])
            self.files.clear()
            self.files.update({str(k): str(v) for k, v in (state.get("files") or {}).items()})

    def _save_session(self) -> None:
        super()._save_session()
        path = self._state_path()
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"todos": self.todos, "files": self.files}, ensure_ascii=False))
        tmp.chmod(0o600)
        tmp.replace(path)

    def reset(self) -> None:
        """Forget the conversation, plan and files, including the saved session."""
        super().reset()
        self.todos = []
        self.files.clear()
        path = self._state_path()
        if path is not None:
            path.unlink(missing_ok=True)
