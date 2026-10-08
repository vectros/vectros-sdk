"""Every SDK feature end to end against a real aios.ko, workers and model.

Needs VECTROS_E2E=1, aios.ko loaded, the storage, tool and LLM workers
running, and a tool-calling model behind the LLM worker.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vectros import (Agent, KernelError, KernelUnavailable, StepLimitReached, VectrosError,
                     tool)

pytestmark = [
    pytest.mark.kernel,
    pytest.mark.skipif(not os.path.exists("/dev/aios") or not os.environ.get("VECTROS_E2E"),
                       reason="set VECTROS_E2E=1 with aios.ko loaded and workers running"),
    pytest.mark.filterwarnings("error:.*predates the loaded kernel"),
]

ROOT = Path(__file__).resolve().parents[1]


def _kernel_tools():
    # /dev/aios lists only the tools a registered agent with a tool core may use.
    with Agent("e2e-list-tools", tools=["placeholder"]) as agent:
        kernel, _ = agent._connect()
        return {entry["name"]: entry for entry in kernel.list_tools()}


@tool
def secret_number() -> int:
    """Returns the secret number."""
    return 4217


# --- running and streaming ---

def test_stream_yields_tokens_then_the_answer():
    with Agent("e2e-stream") as agent:
        events = list(agent.stream("Count from one to five in words."))
    kinds = [event.kind for event in events]
    assert kinds[-1] == "answer" and kinds.count("answer") == 1
    tokens = [event.data for event in events if event.kind == "token"]
    assert len(tokens) >= 2, "the answer did not stream in chunks"
    assert "".join(tokens).strip() == events[-1].data


def test_call_is_run():
    with Agent("e2e-call") as agent:
        assert agent("Reply with the single word: pong").strip()


def test_system_prompt_is_applied():
    with Agent("e2e-system", system="Whatever you are asked, reply with exactly the word BANANA.") as agent:
        assert "banana" in agent.run("What is the capital of France?").lower()


# --- models ---

def test_explicit_allowed_model():
    allowed = Path("/sys/module/aios/parameters/allowed_llm_models").read_text().strip().split(",")
    model = os.environ.get("VECTROS_E2E_MODEL") or allowed[0]
    with Agent("e2e-model", model=model) as agent:
        assert agent.run("Reply with the single word: pong").strip()


def test_model_outside_the_allowlist_is_refused():
    with pytest.raises(KernelError):
        with Agent("e2e-bad-model", model="not-on-the-allowlist") as agent:
            agent.run("hi")


# --- sessions ---

def test_session_survives_a_new_agent_and_reset_forgets_it(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    with Agent("e2e-session", session="s1") as agent:
        agent.run("Remember this: my favourite colour is teal. Reply with OK.")
    path = tmp_path / "vectros" / "sessions" / "e2e-session" / "s1.json"
    assert path.exists() and oct(path.stat().st_mode & 0o777) == "0o600"
    with Agent("e2e-session", session="s1") as agent:
        assert len(agent.history) == 2
        assert "teal" in agent.run("What is my favourite colour? One word.").lower()
        agent.reset()
        assert agent.history == [] and not path.exists()


# --- tools ---

def test_json_tool_protocol():
    with Agent("e2e-json-tools", tools=[secret_number], max_steps=4,
               tool_protocol="json") as agent:
        events = list(agent.stream("Use the secret_number tool, then tell me the value."))
    assert any(e.kind == "tool_call" and e.data["tool"] == "secret_number" for e in events)
    assert "4217" in events[-1].data


def test_tool_error_goes_back_to_the_model():
    @tool
    def read_sensor() -> str:
        """Reads the temperature sensor."""
        raise RuntimeError("sensor offline")

    with Agent("e2e-tool-error", tools=[read_sensor], max_steps=3) as agent:
        events = list(agent.stream("Call read_sensor and tell me what happened."))
    results = [e.data["result"] for e in events if e.kind == "tool_result"]
    assert results and results[0].startswith("error: RuntimeError: sensor offline")
    assert events[-1].kind == "answer"


def test_approval_tool_runs_when_approved():
    seen = []

    @tool(approval=True)
    def launch(target: str) -> str:
        """Launch the rocket at a target."""
        return f"launched at {target}"

    def approve(name, args):
        seen.append((name, args))
        return True

    with Agent("e2e-approve", tools=[launch], max_steps=3, approve=approve) as agent:
        events = list(agent.stream("Call launch with target 'moon'."))
    assert seen and seen[0][0] == "launch"
    results = [e.data["result"] for e in events if e.kind == "tool_result"]
    assert results and results[0].startswith("launched at")


def test_step_limit():
    with pytest.raises(StepLimitReached):
        with Agent("e2e-steps", tools=[secret_number], max_steps=1) as agent:
            agent.run("You must call secret_number before you answer.")


def test_kernel_tool_by_name():
    registry = _kernel_tools()
    name = next((n for n in ("time__get_current_time", "read_file") if n in registry), None)
    if name is None:
        pytest.skip("no suitable kernel tool registered")
    prompt = ("Call time__get_current_time with timezone 'UTC' and tell me the time."
              if name.startswith("time") else
              "Call read_file with path '/etc/hostname' and tell me what it says.")
    with Agent("e2e-kernel-tool", tools=[name], max_steps=4) as agent:
        events = list(agent.stream(prompt))
    results = [e.data["result"] for e in events if e.kind == "tool_result"]
    assert results and not results[0].startswith(("error", "denied")), results


def test_kernel_approval_tool_denied():
    registry = _kernel_tools()
    held = [n for n, entry in registry.items() if entry.get("approval_required")]
    if not held:
        pytest.skip("no kernel tool needs approval")
    name = "fs__create_directory" if "fs__create_directory" in held else held[0]
    events = []
    # A model may keep retrying a denied tool until the step limit; every
    # attempt must still be denied.
    with Agent("e2e-kernel-deny", tools=[name], max_steps=3,
               approve=lambda tool_name, args: False) as agent:
        try:
            for event in agent.stream(f"Call {name} now with any valid arguments."):
                events.append(event)
        except StepLimitReached:
            pass
    results = [e.data["result"] for e in events if e.kind == "tool_result"]
    assert results and all(r.startswith("denied") for r in results), results


def test_unknown_kernel_tool_is_reported():
    with pytest.raises(VectrosError, match="not registered"):
        with Agent("e2e-no-tool", tools=["no_such_tool"]) as agent:
            agent.run("hi")


# --- limits and failures ---

def test_timeout_fails_the_call():
    with pytest.raises(KernelError):
        with Agent("e2e-timeout", timeout=0.001) as agent:
            agent.run("Write a long story about the sea.")


def test_missing_libaios_raises_kernel_unavailable():
    code = ("from vectros import Agent, KernelUnavailable\n"
            "try:\n    Agent('x').run('hi')\n"
            "except KernelUnavailable as e:\n    print('unavailable:', e)\n")
    env = dict(os.environ, VECTROS_LIBAIOS="/nonexistent/libaios.so", PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.startswith("unavailable:"), out.stderr


# --- CLI ---

def _vectros(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, "-m", "vectros", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def test_cli_init_run_and_chat(tmp_path):
    # Separate processes, as a user runs them.
    assert _vectros("init", "e2e-cli", cwd=tmp_path).returncode == 0
    project = tmp_path / "e2e-cli"
    run = _vectros("run", "-C", str(project))
    assert run.returncode == 0 and "6.5" in run.stdout, run.stderr
    chat = _vectros("chat", "-C", str(project), "What is 1.5 + 2? Use the add tool.")
    assert chat.returncode == 0 and "3.5" in chat.stdout, chat.stderr
    assert "-> add" in chat.stderr


def test_python_m_vectros_entry_point(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-m", "vectros", "--help"], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0 and "deploy" in out.stdout
