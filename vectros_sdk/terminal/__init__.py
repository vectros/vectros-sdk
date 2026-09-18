"""AIOS Terminal (plan.md Phase 12a): hybrid command/natural-language REPL."""

from vectros_sdk.terminal.dispatch import DispatchKind, Interpretation, classify
from vectros_sdk.terminal.execution import CommandResult, is_destructive, run_command
from vectros_sdk.terminal.proposal import Proposal, extract_first_proposal, review_and_execute
from vectros_sdk.terminal.repl import Terminal, main

__all__ = [
    "DispatchKind",
    "Interpretation",
    "classify",
    "CommandResult",
    "is_destructive",
    "run_command",
    "Proposal",
    "extract_first_proposal",
    "review_and_execute",
    "Terminal",
    "main",
]
