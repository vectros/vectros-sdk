"""
Operator-typed command execution (ARCH.40/ARCH.42): runs locally through
`$SHELL -c`, output streams directly to the real terminal (inherited stdio,
never captured/buffered), and never goes through AIOS admission -- the
operator already has full host privileges by definition, so routing what
they typed through the kernel would be theatre, not safety.

The one real safety net on this path is `DESTRUCTIVE_VERBS`: a short, fixed
list of verbs whose exact parsed argv must be shown and explicitly confirmed
before running. This is what stops `rm my old files` from silently deleting
something unintended after `dispatch.classify` correctly recognizes `rm` as
a real command (see that module's docstring) -- dispatch decides *what kind*
of thing a line is, this module decides whether a command *of that kind*
needs a human to look at it first.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass

DESTRUCTIVE_VERBS = frozenset({"rm", "dd", "mkfs", "shred", "fdisk", "parted", "kill", "killall"})


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    confirmed: bool
    ran: bool


def is_destructive(command_line: str) -> bool:
    try:
        tokens = shlex.split(command_line)
    except ValueError:
        return False
    if not tokens:
        return False
    verb = os.path.basename(tokens[0])
    if verb in DESTRUCTIVE_VERBS:
        return True
    # A bare shell redirection (`> file`) can destroy data with no verb at
    # all; a cheap substring check is enough for a confirmation prompt (it
    # only ever asks an extra question, never silently skips one).
    return ">" in tokens or any(">" in token for token in tokens)


def run_command(
    command_line: str,
    *,
    confirm=None,
    shell: str | None = None,
) -> CommandResult:
    """Runs `command_line` via `$SHELL -c`, streaming stdio directly.

    Confirms first (via `confirm`, injectable for tests) when the command
    matches `is_destructive`. Returns without running if declined.
    """
    # See proposal.review_and_execute's matching comment: binding a default
    # straight to the `input` builtin would freeze it at import time, before
    # any test patch of `builtins.input` could take effect.
    if confirm is None:
        confirm = input
    if is_destructive(command_line):
        answer = confirm(f"Run destructive command exactly as shown? {command_line!r} [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            return CommandResult(exit_code=0, confirmed=False, ran=False)
        confirmed = True
    else:
        confirmed = True

    shell_path = shell or os.environ.get("SHELL", "/bin/sh")
    completed = subprocess.run([shell_path, "-c", command_line], check=False)
    return CommandResult(exit_code=completed.returncode, confirmed=confirmed, ran=True)
