import json

import pytest

from vectros import Agent, StepLimitReached, VectrosError, tool
from vectros._kernel import SYSCALL_LLM, SYSCALL_STORAGE, SYSCALL_TOOL


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool(approval=True)
def delete_all() -> str:
    """Delete everything."""
    return "deleted"


def test_plain_chat_streams_tokens_and_registers_lazily(fake):
    kernel = fake(replies=["Hello there friend"])
    agent = Agent("hello", model="m1", kernel=kernel)
    assert kernel.registered == []
    events = list(agent.stream("hi"))
    assert "".join(e.data for e in events if e.kind == "token") == "Hello there friend"
    assert events[-1].kind == "answer" and events[-1].data == "Hello there friend"
    assert kernel.registered == ["hello"]
    assert kernel.cores == [(SYSCALL_LLM, "m1")]
    assert kernel.llm_calls[0]["json_mode"] is False
    assert kernel.llm_calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_tool_loop_runs_local_tool_then_answers(fake):
    kernel = fake(replies=[{"tool": "add", "args": {"a": 2.5, "b": 4}},
                           'Sure: {"answer": "6.5"}'])
    agent = Agent("calc", tools=[add], system="Be exact.", kernel=kernel)
    assert agent.run("2.5 + 4?") == "6.5"
    first = kernel.llm_calls[0]
    assert first["json_mode"] is True
    system = first["messages"][0]["content"]
    assert system.startswith("Be exact.") and "- add: Add two numbers." in system
    followup = kernel.llm_calls[1]["messages"]
    assert followup[-1] == {"role": "user", "content": "Tool add returned:\n6.5"}


def test_tool_errors_and_unknown_tools_go_back_to_model(fake):
    @tool
    def boom() -> str:
        """Fails."""
        raise RuntimeError("disk on fire")

    kernel = fake(replies=[{"tool": "boom"}, {"tool": "nope", "args": {}}, {"answer": "gave up"}])
    agent = Agent("t", tools=[boom], kernel=kernel)
    results = [e.data["result"] for e in agent.stream("go") if e.kind == "tool_result"]
    assert results[0] == "error: RuntimeError: disk on fire"
    assert results[1].startswith("error: unknown tool 'nope'")


def test_approval_tool_denied_without_approver(fake):
    kernel = fake(replies=[{"tool": "delete_all", "args": {}}, {"answer": "ok"}])
    agent = Agent("t", tools=[delete_all], kernel=kernel)
    results = [e.data["result"] for e in agent.stream("clean") if e.kind == "tool_result"]
    assert results == ["denied: approval required"]


def test_approval_tool_runs_when_approved(fake):
    seen = []
    kernel = fake(replies=[{"tool": "delete_all", "args": {}}, {"answer": "ok"}])
    agent = Agent("t", tools=[delete_all], kernel=kernel,
                  approve=lambda name, args: seen.append(name) or True)
    results = [e.data["result"] for e in agent.stream("clean") if e.kind == "tool_result"]
    assert results == ["deleted"] and seen == ["delete_all"]


def test_kernel_tools_resolve_from_registry(fake):
    registry = [{"name": "search_web", "approval_required": False,
                 "schema": {"description": "Search the web.",
                            "properties": {"q": {"type": "string"}}}}]
    kernel = fake(replies=[{"tool": "search_web", "args": {"q": "linux"}}, {"answer": "6.x"}],
                  registry=registry, tool_results={"search_web": "Linux 6.x"})
    agent = Agent("r", tools=["search_web", add], kernel=kernel)
    assert agent.run("latest?") == "6.x"
    assert kernel.tool_calls == [("search_web", {"q": "linux"})]
    assert (SYSCALL_TOOL, "") in kernel.cores
    assert "- search_web: Search the web." in kernel.llm_calls[0]["messages"][0]["content"]


def test_missing_kernel_tool_lists_available(fake):
    kernel = fake(registry=[{"name": "fs_read", "approval_required": False, "schema": {}}])
    with pytest.raises(VectrosError, match=r"'search_web' is not registered \(available: fs_read\)"):
        Agent("r", tools=["search_web"], kernel=kernel).run("x")


def test_plain_function_is_rejected(fake):
    def bare(x: int):
        return x

    with pytest.raises(TypeError, match="decorate it with @tool"):
        Agent("r", tools=[bare], kernel=fake()).run("x")


def test_step_limit(fake):
    kernel = fake(replies=[{"tool": "add", "args": {"a": 1, "b": 1}}] * 2)
    with pytest.raises(StepLimitReached):
        Agent("t", tools=[add], max_steps=2, kernel=kernel).run("loop")


def test_session_persists_across_agents(fake, state_home):
    Agent("chat", session="s1", kernel=fake(replies=["Hi Likhin"])).run("I am Likhin")
    path = state_home / "vectros" / "sessions" / "chat" / "s1.json"
    assert json.loads(path.read_text())[-1] == {"role": "assistant", "content": "Hi Likhin"}
    kernel = fake(replies=["You are Likhin"])
    again = Agent("chat", session="s1", kernel=kernel)
    again.run("Who am I?")
    assert kernel.llm_calls[0]["messages"][:2] == [
        {"role": "user", "content": "I am Likhin"},
        {"role": "assistant", "content": "Hi Likhin"}]
    again.reset()
    assert not path.exists() and again.history == []


def test_names_are_validated(fake):
    with pytest.raises(ValueError):
        Agent("../evil", kernel=fake())
    with pytest.raises(ValueError):
        Agent("ok", session="a/b", kernel=fake())


def test_storage_sets_up_core_once(fake):
    kernel = fake()
    agent = Agent("s", kernel=kernel)
    agent.storage.write("notes.txt", "hi")
    assert agent.storage.read("notes.txt") == b"hi"
    assert kernel.cores.count((SYSCALL_STORAGE, "")) == 1


def test_close_unregisters(fake):
    kernel = fake(replies=["x"])
    with Agent("c", kernel=kernel) as agent:
        agent.run("x")
        agent_id = agent.id
    assert kernel.unregistered == [agent_id]
