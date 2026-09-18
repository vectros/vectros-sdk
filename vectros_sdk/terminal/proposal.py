"""
Agent-proposed commands (ARCH.40, TERM.6): a model reply never executes
anything on its own. The real backend has no structured tool-calling channel
today (`vectros_sdk.client.real_kernel._llm` raises `RealBackendUnsupported`
for `tool_use`/`operate_file`) -- so a proposed command can only ever arrive
as a fenced code block inside the model's plain-text reply, by an explicit
convention this module owns end to end (see `PROPOSE_SYSTEM_PROMPT` for the
exact instruction given to the model). The fence language is one of a fixed,
named set (`propose`, `bash`, `sh`, `shell`) rather than "any fence" -- real
small local models reach for the ordinary `bash`/`sh` markdown convention
even when told to use `propose`, so the set matches actual observed model
behavior instead of a convention no model follows; it is still a closed,
explicit list, never inferred from content.

Only the FIRST matching fenced block in a reply is ever parsed, shown, or
run. Any further block is discarded unread. This is the direct, structural
defense against a model trying to smuggle a second command past one the
human already approved (TERM.14): there is no code path that ever looks at
block number two.

Approval is local and locally audited (plan.md Phase 12a: kernel-side
approval is Phase 22 CTL.10, not this phase) -- one JSON line per decision,
appended to `AUDIT_LOG_PATH`.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vectros_sdk.terminal.execution import CommandResult, run_command

PROPOSE_SYSTEM_PROMPT = (
    "You are an assistant embedded in a Linux terminal for the AIOS agent "
    "operating system. You cannot execute commands yourself. When running a "
    "shell command would help, respond with exactly one fenced code block "
    "containing only that command, for example:\n"
    "```bash\ndate\n```\n"
    "A human will see exactly that command and must approve it before "
    "anything runs -- nothing executes automatically. Only ever include one "
    "command per reply."
)

_PROPOSE_FENCE_LANGUAGES = ("propose", "bash", "sh", "shell")
_PROPOSE_BLOCK = re.compile(
    r"```(?:" + "|".join(_PROPOSE_FENCE_LANGUAGES) + r")\s*\n(.*?)\n```", re.DOTALL
)


def _default_audit_log_path() -> Path:
    override = os.environ.get("AIOS_TERMINAL_AUDIT_LOG")
    if override:
        return Path(override)
    return Path.home() / ".aios" / "terminal_audit.jsonl"


AUDIT_LOG_PATH = _default_audit_log_path()


@dataclass(frozen=True)
class Proposal:
    command: str
    # The model's own prose before the fenced block -- user-facing;
    # everything from the fence onward is an internal protocol detail
    # between this terminal and the model, never shown raw.
    prose_before: str


def extract_first_proposal(reply_text: str) -> Optional[Proposal]:
    """Returns only the first matching fenced block, ignoring any later
    one -- see the module docstring for exactly which fence languages
    count and why."""
    match = _PROPOSE_BLOCK.search(reply_text)
    if match is None:
        return None
    command = match.group(1).strip()
    if not command:
        return None
    return Proposal(command=command, prose_before=reply_text[: match.start()].strip())


def _append_audit_line(command: str, decision: str, exit_code: Optional[int]) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_ms": int(time.time() * 1000),
        "command": command,
        "decision": decision,
        "exit_code": exit_code,
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def review_and_execute(proposal: Proposal, *, confirm=None) -> CommandResult:
    """Shows the exact proposed argv, requires approval, then runs it
    through the same local `$SHELL -c` path operator-typed commands use --
    the only difference from that path is this explicit approval gate."""
    # `confirm=None` rather than `confirm=input`: a default bound directly to
    # the `input` builtin is captured once at import time and never sees a
    # test's `unittest.mock.patch("builtins.input", ...)` again -- looking it
    # up here, at call time, is what makes patching actually work.
    if confirm is None:
        confirm = input
    print(f"Agent proposes to run: {proposal.command!r}")
    answer = confirm("Approve and run this command? [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        _append_audit_line(proposal.command, "declined", None)
        return CommandResult(exit_code=0, confirmed=False, ran=False)

    # Already approved -- `run_command`'s own destructive-verb confirmation
    # would otherwise ask a second, redundant question for e.g. `rm`.
    result = run_command(proposal.command, confirm=lambda _prompt: "y")
    _append_audit_line(proposal.command, "approved", result.exit_code)
    return result
