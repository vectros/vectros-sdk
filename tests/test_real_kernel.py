"""End-to-end against a real aios.ko with LLM and storage workers running."""

import os

import pytest

from vectros import Agent, tool

pytestmark = [
    pytest.mark.kernel,
    pytest.mark.skipif(not os.path.exists("/dev/aios") or not os.environ.get("VECTROS_E2E"),
                       reason="set VECTROS_E2E=1 with aios.ko loaded and workers running"),
]


def test_plain_run():
    with Agent("e2e-hello") as agent:
        assert agent.run("Reply with the single word: pong").strip()


def test_local_tool_round_trip():
    @tool
    def secret_number() -> int:
        """Returns the secret number."""
        return 4217

    with Agent("e2e-tools", tools=[secret_number], max_steps=4) as agent:
        assert "4217" in agent.run("Call secret_number and tell me the value.")


def test_storage_round_trip():
    with Agent("e2e-storage") as agent:
        agent.storage.write("note.txt", "hello")
        assert agent.storage.read("note.txt") == b"hello"
