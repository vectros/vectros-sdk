"""
The AIOS Terminal REPL (plan.md Phase 12a).

One prompt, hybrid dispatch (ARCH.39): a resolved interpretation is always
shown before anything runs. Operator-typed lines execute locally with no
AIOS admission (ARCH.40/42, `execution.py`). Natural-language lines go to a
real model over the real kernel protocol (ARCH.38, `vectros_sdk.client.
client.AIOSClient`) and any proposed command in the reply requires explicit
approval before it ever runs (`proposal.py`).

Session as an AIOS Context (ARCH.45): one `AIOSClient` and one growing
`messages` list live for the whole REPL process. `real_kernel.py`'s `_llm`
already keeps one real `ContextManager`-backed context per agent for the
process's lifetime and appends the newest message into it on every call --
so holding a persistent client and resending the full history each turn (the
same shape any chat client already uses) is what makes this REPL's session
a real AIOS Context, not new plumbing.

Delegates, never reimplements (ARCH.44): line editing and history come from
the stdlib `readline` module (loaded for its side effect on `input()`);
command execution delegates to the user's own `$SHELL`.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import readline  # noqa: F401  (side effect: input() gets history/editing)
except ImportError:  # pragma: no cover - readline is unavailable on some platforms
    pass

from vectros_sdk.client.client import AIOSClient
from vectros_sdk.terminal.dispatch import DispatchKind, classify
from vectros_sdk.terminal.execution import run_command
from vectros_sdk.terminal.proposal import (
    PROPOSE_SYSTEM_PROMPT,
    extract_first_proposal,
    review_and_execute,
)

PROMPT = "aios> "


class Terminal:
    """One REPL session: one agent identity, one client, one growing
    conversation. Constructing this performs no I/O; nothing talks to the
    kernel until the first natural-language turn."""

    def __init__(self, *, agent_name: str, socket_path: str, model: str):
        self.client = AIOSClient(agent_name=agent_name, socket_path=socket_path)
        self.model = model
        self.messages: list[dict] = [{"role": "system", "content": PROPOSE_SYSTEM_PROMPT}]

    def handle_line(self, line: str) -> None:
        interpretation = classify(line)
        if interpretation.kind is DispatchKind.COMMAND:
            print(f"[running command] {interpretation.text}")
            run_command(interpretation.text)
            return

        print(f"[asking model] {interpretation.text}")
        self.messages.append({"role": "user", "content": interpretation.text})
        try:
            response = self.client.llm.chat(
                messages=self.messages,
                llms=[{"name": self.model}],
            )
        except Exception as error:  # noqa: BLE001 - a live model/kernel call has
            # many distinct real failure modes (RealBackendUnsupported,
            # ExecutionProtocolError, AIOSKernelError, a dead socket...); the
            # REPL's job is to report whichever one occurred and keep running,
            # not to crash the whole session over one bad turn.
            self.messages.pop()  # no assistant reply exists for this turn
            print(f"[error] {error}")
            return
        reply_text = response.response_message or ""
        self.messages.append({"role": "assistant", "content": reply_text})

        proposal = extract_first_proposal(reply_text)
        if proposal is None:
            print(reply_text)
            return
        # Show the model's own words up to the proposal, then the approval
        # gate -- never the raw fenced block, which is an internal protocol
        # detail between this terminal and the model, not user-facing text.
        if proposal.prose_before:
            print(proposal.prose_before)
        review_and_execute(proposal)

    def run(self) -> None:
        while True:
            try:
                line = input(PROMPT)
            except EOFError:
                print()
                return
            except KeyboardInterrupt:
                print()
                continue
            if line.strip() in ("exit", "quit"):
                return
            if not line.strip():
                continue
            self.handle_line(line)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aios-terminal", description="AIOS Terminal (Phase 12a)")
    parser.add_argument(
        "--socket",
        default=os.environ.get("AIOS_TERMINAL_SOCKET"),
        help="Path to a real, standing kernel server's Unix socket (aiosctl serve-kernel). "
        "Defaults to $AIOS_TERMINAL_SOCKET.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AIOS_TERMINAL_MODEL"),
        help="Exact model name the kernel server was started with "
        "(aiosctl serve-kernel ... --ollama-model <name>). Defaults to $AIOS_TERMINAL_MODEL.",
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("AIOS_TERMINAL_AGENT", "terminal"),
        help="Agent identity suffix for this session -- must be the SAME suffix "
        "the server's own $AIOS_AGENT_ID was started with, minus its 'agent_' "
        "prefix (e.g. server started with AIOS_AGENT_ID=agent_terminal -> pass "
        "'terminal' here; tenant_permissions() grants are namespaced by this "
        "exact suffix). Defaults to $AIOS_TERMINAL_AGENT or 'terminal'.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.socket or not args.model:
        print(
            "aios-terminal requires --socket/$AIOS_TERMINAL_SOCKET and "
            "--model/$AIOS_TERMINAL_MODEL naming a real standing kernel server "
            "(start one with: aiosctl serve-kernel <socket> <telemetry-dir> "
            "--ollama-model <model>).",
            file=sys.stderr,
        )
        return 2
    terminal = Terminal(agent_name=args.agent_id, socket_path=args.socket, model=args.model)
    terminal.run()
    return 0
