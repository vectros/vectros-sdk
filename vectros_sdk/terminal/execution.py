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

TERM.7: real TTY passthrough for interactive programs (ARCH.43), "no PTY
multiplexer" -- the child gets this process's own real controlling-terminal
file descriptors directly (already true of the plain `subprocess.run` path
below, which never redirects stdio), not a virtual pty pair we shuttle bytes
through ourselves. What plain `subprocess.run` gets wrong for a genuinely
interactive program (`vim`, `less`, `ssh`, a REPL, ...): the child is never
given its own process group, so it stays in *this* process's foreground
process group -- meaning the terminal driver delivers Ctrl-C/Ctrl-Z/Ctrl-\\
to this wrapper process too, and `subprocess.run`'s own KeyboardInterrupt
handling then kills the child outright instead of letting it handle the
signal itself (mysql interrupting one query, vim ignoring it, ...). The real
fix is standard job control, exactly what every real shell does when it runs
a foreground pipeline: give the child (here, the `$SHELL -c` process; any
program *it* execs inherits the same group automatically) its own process
group, hand the terminal's foreground status to that group for the duration
of the run, then hand it back to this wrapper afterward so the REPL's own
next `input()` call is not left talking to a terminal that thinks a
now-exited child still owns it.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass

DESTRUCTIVE_VERBS = frozenset({"rm", "dd", "mkfs", "shred", "fdisk", "parted", "kill", "killall"})

# TERM.7: a real, deliberately narrow heuristic -- true only when the
# command's own first token's basename is one of these well-known,
# TTY-requiring programs. This is not a general interactive-program
# detector: a wrapper script, a shell alias, or a real interactive program
# not on this fixed list runs through the ordinary `subprocess.run` path
# below instead, which remains correct for the overwhelming majority of
# commands (anything that does not itself read/write the controlling
# terminal in a job-control-sensitive way). A false negative here costs
# nothing new (it is exactly today's pre-TERM.7 behavior); a false positive
# only costs one extra, harmless process-group handoff. Extend this list as
# real gaps are found rather than trying to make it exhaustive up front.
INTERACTIVE_PROGRAMS = frozenset(
    {
        "vim",
        "vi",
        "nvim",
        "emacs",
        "nano",
        "pico",
        "less",
        "more",
        "man",
        "top",
        "htop",
        "watch",
        "ssh",
        "mosh",
        "mysql",
        "psql",
        "sqlite3",
        "redis-cli",
        "python",
        "python3",
        "ipython",
        "node",
        "irb",
        "R",
        "tmux",
        "screen",
        "vipw",
        "visudo",
        "crontab",
        "fdisk",
        "cfdisk",
        "parted",
    }
)


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


def is_interactive(command_line: str) -> bool:
    """True when `command_line`'s own first token names a known
    TTY-requiring program -- see `INTERACTIVE_PROGRAMS`'s own doc comment
    for exactly what this does and does not cover."""
    try:
        tokens = shlex.split(command_line)
    except ValueError:
        return False
    if not tokens:
        return False
    return os.path.basename(tokens[0]) in INTERACTIVE_PROGRAMS


def _run_with_terminal_control(argv: list[str]) -> int:
    """Runs `argv` as the real foreground process group of this process's
    own controlling terminal for the duration of the run (TERM.7/ARCH.43),
    then hands the terminal back. See this module's own top-of-file doc
    comment for why this differs from plain `subprocess.run`.
    """
    # A real, deliberately raw-fd check, not `sys.stdin.isatty()`: something
    # further up the process (a test harness's own capture layer, a
    # `readline`-adjacent wrapper, ...) can replace the *Python-level*
    # `sys.stdin` object with something whose `isatty()` lies, while the
    # real, inherited OS file descriptor 0 this function actually operates
    # on (`os.tcsetpgrp`, below) is unaffected by that -- confirmed live
    # while building this: under this project's own pytest runner, checking
    # `sys.stdin.isatty()` reported `False` against a real pty this test
    # itself had just attached as fd 0, which would have silently skipped
    # job control entirely inside a real interactive terminal session too,
    # for any caller with the same kind of stdin wrapping.
    if not os.isatty(0):
        # No real controlling terminal to hand off (piped/captured stdio,
        # e.g. when input is redirected) -- job control has nothing to
        # attach to, so there is nothing this path can do that the plain
        # path below does not already do correctly.
        return subprocess.run(argv, check=False).returncode

    terminal_fd = 0
    own_pgrp = os.getpgrp()

    # `process_group=0` (Python 3.11+) asks `subprocess` to put the child in
    # a fresh process group (pgid == the child's own pid) itself, via the
    # safe internal fork/exec path. The obvious alternative on older
    # interpreters, `preexec_fn=os.setpgrp`, works too (verified directly,
    # not assumed) but the stdlib's own docs still warn it is not fork-safe
    # in a process with other threads -- a real possibility here (this
    # REPL's own dependencies can pull in gRPC/asyncio-based transports),
    # so it is used only where the safer option does not exist.
    if sys.version_info >= (3, 11):
        process = subprocess.Popen(argv, process_group=0)
    else:
        process = subprocess.Popen(argv, preexec_fn=os.setpgrp)  # noqa: PLW1509

    # Handing the terminal to a group other than the current foreground one
    # is itself a terminal-control operation the driver can raise SIGTTOU
    # over if this wrapper is ever not the foreground group when it calls
    # `tcsetpgrp` (it always is, here) -- ignoring it around the call is the
    # standard, defensive pattern every real job-control shell uses.
    def _tcsetpgrp_ignoring_sigttou(pgrp: int) -> None:
        previous = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
        try:
            os.tcsetpgrp(terminal_fd, pgrp)
        finally:
            signal.signal(signal.SIGTTOU, previous)

    _tcsetpgrp_ignoring_sigttou(process.pid)
    try:
        return process.wait()
    finally:
        # Restore this wrapper as the foreground group unconditionally --
        # even if the child was killed or `wait()` raised -- so the REPL's
        # next `input()` call is not left talking to a terminal that still
        # thinks the (now-gone) child's group owns it.
        _tcsetpgrp_ignoring_sigttou(own_pgrp)


def run_command(
    command_line: str,
    *,
    confirm=None,
    shell: str | None = None,
) -> CommandResult:
    """Runs `command_line` via `$SHELL -c`, streaming stdio directly.

    Confirms first (via `confirm`, injectable for tests) when the command
    matches `is_destructive`. Returns without running if declined. When
    `is_interactive` matches, runs via `_run_with_terminal_control` (TERM.7)
    instead of the plain path, so the program gets real, unmultiplexed
    control of the terminal for its own job-control signals.
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
    argv = [shell_path, "-c", command_line]
    if is_interactive(command_line):
        exit_code = _run_with_terminal_control(argv)
    else:
        exit_code = subprocess.run(argv, check=False).returncode
    return CommandResult(exit_code=exit_code, confirmed=confirmed, ran=True)
