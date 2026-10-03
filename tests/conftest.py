import json

import pytest


class FakeKernel:
    """Stands in for vectros._kernel.Kernel; replies come from a script."""

    def __init__(self, replies=(), registry=(), tool_results=None):
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
                   timeout_ms=0, span=None):
        self.llm_calls.append({"messages": messages, "model": model, "json_mode": json_mode})
        reply = self.replies.pop(0)
        if isinstance(reply, dict):
            reply = json.dumps(reply)
        for index in range(0, len(reply), 4):
            yield reply[index:index + 4]
        return reply

    def tool(self, agent_id, name, args, timeout_ms=0, approve=None, span=None):
        self.tool_calls.append((name, args))
        result = self.tool_results[name]
        if isinstance(result, Exception):
            raise result
        return result

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
