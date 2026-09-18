"""
Hybrid command/natural-language dispatch for the AIOS Terminal (ARCH.39).

Pure classification: given one input line, decide whether it is a shell
command (delegated to `$SHELL -c`, see `execution.py`) or natural language
(sent to the model, see `repl.py`). No I/O happens here, which is what makes
TERM.13's dispatch-table test possible without a shell or a model.

Rule: `!`/`?` force an interpretation; otherwise the first whitespace-split
token is checked against `PATH` (`shutil.which`) — if it resolves, the whole
line is a command, else it is natural language. This is deliberately simple
and deliberately visible (the caller always shows the resolved
`Interpretation` before doing anything, per ARCH.39) rather than trying to be
clever about intent: `rm my old files` dispatches as the command `rm my old
files` (because `rm` *is* in `PATH`) precisely as plan.md's own motivating
example says it should — the safety net for that case is
`execution.DESTRUCTIVE_VERBS`, not smarter dispatch here.
"""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from enum import Enum


class DispatchKind(Enum):
    COMMAND = "command"
    NATURAL_LANGUAGE = "natural_language"


@dataclass(frozen=True)
class Interpretation:
    kind: DispatchKind
    # The exact argv `execution.run_command` would pass to `$SHELL -c`, or
    # the natural-language text to send to the model — never both.
    text: str
    forced: bool = False


def classify(line: str, *, which=shutil.which) -> Interpretation:
    """Classifies one raw input line. `which` is injectable for tests."""
    stripped = line.strip()
    if not stripped:
        return Interpretation(DispatchKind.NATURAL_LANGUAGE, stripped, forced=False)

    if stripped.startswith("!"):
        return Interpretation(DispatchKind.COMMAND, stripped[1:].strip(), forced=True)
    if stripped.startswith("?"):
        return Interpretation(DispatchKind.NATURAL_LANGUAGE, stripped[1:].strip(), forced=True)

    try:
        tokens = shlex.split(stripped)
    except ValueError:
        # Unbalanced quotes etc. -- not a parseable command line, so treat it
        # as natural language rather than guessing at a broken argv.
        return Interpretation(DispatchKind.NATURAL_LANGUAGE, stripped, forced=False)

    if tokens and which(tokens[0]) is not None:
        return Interpretation(DispatchKind.COMMAND, stripped, forced=False)
    return Interpretation(DispatchKind.NATURAL_LANGUAGE, stripped, forced=False)
