import json

import pytest

from vectros import ToolDenied


class FakeKernel:
    """Stands in for vectros._kernel.Kernel; replies come from a script."""

    def __init__(self, replies=(), registry=(), tool_results=None, client_tools=True):
        self.supports_client_tools = client_tools
        self.client_calls = []
        self.replies = list(replies)
        self.registry = list(registry)
        self.tool_results = tool_results or {}
        self.llm_calls = []
        self.tool_calls = []
        self.cores = []
        self.registered = []
        self.unregistered = []
        self.files = {}

    def register(self, name):
        self.registered.append(name)
        return 1000 + len(self.registered)

    def unregister(self, agent_id):
        self.unregistered.append(agent_id)

    def close(self):
        pass

    def setup_core(self, agent_id, core, pstr=""):
        self.cores.append((core, pstr))

    def trace_enabled(self, agent_id):
        return False

    def llm_stream(self, agent_id, messages, model=None, json_mode=False,
                   timeout_ms=0, span=None, tools=None):
        self.llm_calls.append({"messages": [dict(m) for m in messages], "model": model,
                               "json_mode": json_mode, "tools": tools})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        # Like the worker: only content streams; the result is the full reply.
        streamed = reply.get("content", "") if isinstance(reply, dict) and \
            reply.get("aios_llm_result") == 1 else reply
        if isinstance(streamed, dict):
            streamed = json.dumps(streamed)
        for index in range(0, len(streamed), 4):
            yield streamed[index:index + 4]
        return json.dumps(reply) if isinstance(reply, dict) else reply

    def tool(self, agent_id, name, args, timeout_ms=0, approve=None, span=None):
        self.tool_calls.append((name, args))
        result = self.tool_results[name]
        if isinstance(result, Exception):
            raise result
        return result

    def client_tool(self, agent_id, name, args, run, approval=False, approve=None,
                    timeout_ms=0, span=None):
        """Mimics the kernel: held calls need approve(), else the owner denies."""
        self.client_calls.append((name, args, approval))
        if approval and not (approve and approve(name, args)):
            raise ToolDenied(name, "denied or cancelled by the owner")
        return run()

    def list_tools(self):
        return self.registry

    def storage_write(self, agent_id, path, data):
        self.files[path] = data

    def storage_read(self, agent_id, path):
        return self.files[path]


@pytest.fixture
def fake():
    return FakeKernel


@pytest.fixture(autouse=True)
def state_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "state"
