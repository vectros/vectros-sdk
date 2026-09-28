import json
from types import SimpleNamespace

import pytest

from vectros_sdk.terminal.structured_agent import StructuredTerminalAgent, parse_action
from vectros_sdk.transport.execution_protocol import ExecutionProtocolError
from vectros_sdk.client.real_kernel import _tool
from vectros_sdk.terminal.execution import CommandResult
from vectros_sdk.terminal import structured_agent as agent_module
from vectros_sdk.terminal import proposal as proposal_module


@pytest.mark.parametrize(
    "text",
    [
        "```json\n{}\n```",
        '{"action":"tool","tool":"bash","arguments":{"command":"ls"}}',
        '{"action":"tool","tool":"pwd","arguments":{"path":"/"}}',
        '{"action":"tool","tool":"list_dir","arguments":{"path":7}}',
        '{"action":"final","answer":"ok","tool":"pwd"}',
    ],
)
def test_rejects_unstructured_or_unapproved_actions(text):
    with pytest.raises(ValueError):
        parse_action(text)


def test_structured_bash_action_accepts_any_exact_command_string():
    action = parse_action(json.dumps({"action": "bash", "command": 'find . -name "*.md" | wc -l'}))
    assert action["command"] == 'find . -name "*.md" | wc -l'


def test_agent_calls_approved_kernel_tool_and_returns_actual_result():
    class FakeClient:
        def __init__(self):
            self.prompts = []
            self.calls = []
            self.pending = True
            self.llm = self
            self.tool = self
            self.control = self

        def chat_json(self, *, messages, response_format, llms):
            self.prompts.append(messages[-1]["content"])
            return SimpleNamespace(
                response_message='{"action":"final","answer":"The workspace has Cargo.toml."}'
            )

        def call(self, calls):
            self.calls.append(calls)
            if len(self.calls) == 1:
                raise ExecutionProtocolError("held", code="pending_approval")
            return SimpleNamespace(response_message="Cargo.toml\ncrates/")

        def list_pending_approvals(self):
            if self.pending:
                self.pending = False
                return [{"approval_id": "approval_1"}]
            return []

    fake = FakeClient()
    events = []
    agent = StructuredTerminalAgent(
        socket_path="/tmp/test.sock", model="model", client=fake,
        on_event=events.append, sleep=lambda _: None,
    )
    result = agent.run("list the files")
    assert fake.calls == [
        [{"name": "list_dir", "parameters": {"path": "."}}],
        [{"name": "list_dir", "parameters": {"path": "."}}],
    ]
    assert result["tool_results"] == [{"tool": "list_dir", "output": "Cargo.toml\ncrates/"}]
    assert "Cargo.toml" in fake.prompts[0]
    assert any("Tool result (list_dir)" in event for event in events)


def test_sdk_maps_list_dir_to_registered_kernel_resource():
    class FakeProtocol:
        def __init__(self):
            self.operation = None

        def execute(self, request_id, operation):
            self.operation = operation
            return {"ToolResult": [["output", {"String": "Cargo.toml"}]]}

    protocol = FakeProtocol()
    response = _tool(
        SimpleNamespace(
            agent_name="terminal",
            tool_calls=[{"name": "list_dir", "parameters": {"path": "."}}],
        ),
        protocol,
    )
    assert response["response_message"] == "Cargo.toml"
    assert "res_tool_list_dir_terminal" in str(protocol.operation)


def test_invalid_model_actions_never_reach_tool_client():
    class FakeClient:
        def __init__(self):
            self.llm = self
            self.tool = self
            self.calls = 0

        def chat_json(self, **kwargs):
            return SimpleNamespace(response_message='{"action":"tool","tool":"bash","arguments":{}}')

        def call(self, calls):
            self.calls += 1
            raise AssertionError("unapproved command reached tool client")

    fake = FakeClient()
    agent = StructuredTerminalAgent(socket_path="/tmp/test.sock", model="model", client=fake)
    with pytest.raises(ValueError):
        agent.run("run arbitrary bash")
    assert fake.calls == 0


def test_non_listing_task_uses_model_json_tool_action():
    class FakeClient:
        def __init__(self):
            self.llm = self
            self.tool = self
            self.prompts = []
            self.calls = []

        def chat_json(self, *, messages, **kwargs):
            self.prompts.append(messages[-1]["content"])
            if len(self.prompts) == 1:
                return SimpleNamespace(response_message='{"action":"tool","tool":"uppercase","arguments":{"input":"aios"}}')
            return SimpleNamespace(response_message='{"action":"final","answer":"AIOS"}')

        def call(self, calls):
            self.calls.append(calls)
            return SimpleNamespace(response_message="AIOS")

    fake = FakeClient()
    agent = StructuredTerminalAgent(socket_path="/tmp/test.sock", model="model", client=fake, on_event=lambda _: None)
    result = agent.run("transform aios to capitals")
    assert fake.calls == [[{"name": "uppercase", "parameters": {"input": "aios"}}]]
    assert result["answer"] == "AIOS"


def test_scripted_uppercase_task_is_an_agent_tool_call():
    class FakeClient:
        def __init__(self):
            self.llm = self
            self.tool = self
            self.calls = []

        def chat_json(self, **kwargs):
            return SimpleNamespace(response_message='{"action":"final","answer":"VECTROS KERNEL"}')

        def call(self, calls):
            self.calls.append(calls)
            return SimpleNamespace(response_message="VECTROS KERNEL")

    fake = FakeClient()
    result = StructuredTerminalAgent(
        socket_path="/tmp/test.sock", model="model", client=fake,
        on_event=lambda _: None,
    ).run("uppercase vectros kernel")
    assert fake.calls == [[{"name": "uppercase", "parameters": {"input": "vectros kernel"}}]]
    assert result["tool_results"][0]["output"] == "VECTROS KERNEL"


def test_model_bash_action_uses_exact_local_approval_path(monkeypatch):
    command = 'find . -name "*.md" | wc -l'
    seen = []

    class FakeClient:
        def __init__(self):
            self.llm = self
            self.tool = self

        def chat_json(self, **kwargs):
            return SimpleNamespace(response_message=json.dumps({"action": "bash", "command": command}))

        def call(self, calls):
            raise AssertionError("Bash must not be sent as a kernel fixture tool")

    def approved(proposal):
        seen.append(proposal.command)
        return CommandResult(exit_code=0, confirmed=True, ran=True)

    monkeypatch.setattr(agent_module, "review_and_execute", approved)
    agent = StructuredTerminalAgent(
        socket_path="/tmp/test.sock", model="model", client=FakeClient(),
        on_event=lambda _: None,
    )
    result = agent.run("find the total markdown files")
    assert seen == [command]
    assert "status 0" in result["answer"]


def test_declined_bash_proposal_is_not_reported_as_run(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.llm = self

        def chat_json(self, **kwargs):
            return SimpleNamespace(response_message='{"action":"bash","command":"rm -rf important"}')

    monkeypatch.setattr(
        agent_module, "review_and_execute",
        lambda proposal: CommandResult(exit_code=0, confirmed=False, ran=False),
    )
    agent = StructuredTerminalAgent(socket_path="/tmp/test.sock", model="model", client=FakeClient(), on_event=lambda _: None)
    result = agent.run("delete important")
    assert result["answer"] == "Command was not approved and did not run."


def test_operator_bang_runs_exact_bash_command_without_model(monkeypatch):
    commands = []

    def run(command):
        commands.append(command)
        return CommandResult(exit_code=0, confirmed=True, ran=True)

    monkeypatch.setattr(agent_module, "run_command", run)
    agent = StructuredTerminalAgent(
        socket_path="/tmp/test.sock", model="model", client=object(),
        on_event=lambda _: None,
    )
    result = agent.run('!find . -name "*.md" | wc -l')
    assert commands == ['find . -name "*.md" | wc -l']
    assert "status 0" in result["answer"]


def test_approved_model_pipeline_streams_real_output_and_is_audited(monkeypatch, tmp_path, capfd):
    command = "printf 'one\\ntwo\\n' | wc -l"

    class FakeClient:
        def __init__(self):
            self.llm = self

        def chat_json(self, **kwargs):
            return SimpleNamespace(response_message=json.dumps({"action": "bash", "command": command}))

    monkeypatch.setattr("builtins.input", lambda _: "y")
    monkeypatch.setattr(proposal_module, "AUDIT_LOG_PATH", tmp_path / "terminal_audit.jsonl")
    agent = StructuredTerminalAgent(
        socket_path="/tmp/test.sock", model="model", client=FakeClient(),
        on_event=lambda _: None,
    )
    result = agent.run("count lines")
    assert "2" in capfd.readouterr().out
    assert "status 0" in result["answer"]
    audit = json.loads((tmp_path / "terminal_audit.jsonl").read_text().splitlines()[0])
    assert audit["command"] == command
    assert audit["decision"] == "approved"
