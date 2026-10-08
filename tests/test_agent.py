import json

import pytest

from vectros import Agent, KernelError, StepLimitReached, VectrosError, tool
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
    agent = Agent("calc", tools=[add], system="Be exact.", tool_protocol="json", kernel=kernel)
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
    agent = Agent("t", tools=[boom], tool_protocol="json", kernel=kernel)
    results = [e.data["result"] for e in agent.stream("go") if e.kind == "tool_result"]
    assert results[0] == "error: RuntimeError: disk on fire"
    assert results[1].startswith("error: unknown tool 'nope'")


def test_approval_tool_denied_without_approver(fake):
    kernel = fake(replies=[{"tool": "delete_all", "args": {}}, {"answer": "ok"}])
    agent = Agent("t", tools=[delete_all], tool_protocol="json", kernel=kernel)
    results = [e.data["result"] for e in agent.stream("clean") if e.kind == "tool_result"]
    assert results == ["denied: denied or cancelled by the owner"]


def test_approval_tool_runs_when_approved(fake):
    seen = []
    kernel = fake(replies=[{"tool": "delete_all", "args": {}}, {"answer": "ok"}])
    agent = Agent("t", tools=[delete_all], kernel=kernel, tool_protocol="json",
                  approve=lambda name, args: seen.append(name) or True)
    results = [e.data["result"] for e in agent.stream("clean") if e.kind == "tool_result"]
    assert results == ["deleted"] and seen == ["delete_all"]


def test_kernel_tools_resolve_from_registry(fake):
    registry = [{"name": "search_web", "approval_required": False,
                 "schema": {"description": "Search the web.",
                            "properties": {"q": {"type": "string"}}}}]
    kernel = fake(replies=[{"tool": "search_web", "args": {"q": "linux"}}, {"answer": "6.x"}],
                  registry=registry, tool_results={"search_web": "Linux 6.x"})
    agent = Agent("r", tools=["search_web", add], tool_protocol="json", kernel=kernel)
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
        Agent("t", tools=[add], max_steps=2, tool_protocol="json", kernel=kernel).run("loop")


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


def envelope(content="", calls=()):
    return {"aios_llm_result": 1, "content": content, "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(calls)]}


def test_native_tool_calls_run_through_kernel_and_feed_back(fake):
    kernel = fake(replies=[envelope("Let me add.", [("add", {"a": 2, "b": 3}), ("add", {"a": 1, "b": 1})]),
                           envelope("The sums are 5 and 2.")])
    agent = Agent("calc", tools=[add], system="Be exact.", kernel=kernel)
    events = list(agent.stream("2+3 and 1+1?"))
    assert events[-1].data == "The sums are 5 and 2."
    first = kernel.llm_calls[0]
    assert first["json_mode"] is False
    assert first["tools"] == [{"type": "function", "function": {
        "name": "add", "description": "Add two numbers.", "parameters": add.parameters}}]
    assert first["messages"][0] == {"role": "system", "content": "Be exact."}
    followup = kernel.llm_calls[1]["messages"]
    assert followup[-3]["role"] == "assistant" and len(followup[-3]["tool_calls"]) == 2
    assert followup[-2:] == [{"role": "tool", "tool_call_id": "c0", "content": "5"},
                             {"role": "tool", "tool_call_id": "c1", "content": "2"}]
    assert kernel.client_calls == [("add", {"a": 2, "b": 3}, False), ("add", {"a": 1, "b": 1}, False)]
    assert "".join(e.data for e in events if e.kind == "token").startswith("Let me add.")


def test_empty_reply_after_tool_results_is_retried_once(fake):
    empty = KernelError(5, "LLM backend returned no content (finish_reason=stop)")
    kernel = fake(replies=[envelope("", [("add", {"a": 2, "b": 3})]), empty, envelope("It is 5.")])
    agent = Agent("calc", tools=[add], kernel=kernel)
    assert agent.run("2+3?") == "It is 5."
    retry = kernel.llm_calls[2]["messages"]
    assert retry[-1]["role"] == "user" and "final answer" in retry[-1]["content"]
    assert agent.history == [{"role": "user", "content": "2+3?"},
                             {"role": "assistant", "content": "It is 5."}]


def test_empty_reply_without_tool_results_is_an_error(fake):
    empty = KernelError(5, "LLM backend returned no content (finish_reason=stop)")
    with pytest.raises(KernelError, match="no content"):
        Agent("t", tools=[add], kernel=fake(replies=[empty])).run("hi")


def test_native_plain_text_reply_is_the_answer(fake):
    kernel = fake(replies=["No tools needed."])
    assert Agent("t", tools=[add], kernel=kernel).run("hi") == "No tools needed."


def test_native_bad_arguments_are_reported(fake):
    bad = {"aios_llm_result": 1, "content": "", "tool_calls": [
        {"id": "x", "function": {"name": "add", "arguments": "{not json"}}]}
    kernel = fake(replies=[bad, envelope("sorry")])
    agent = Agent("t", tools=[add], kernel=kernel)
    results = [e.data["result"] for e in agent.stream("go") if e.kind == "tool_result"]
    assert results == ["error: arguments must be a JSON object"]


def test_local_tools_fall_back_in_process_on_old_kernels(fake):
    kernel = fake(replies=[envelope("", [("delete_all", {})]), envelope("done")], client_tools=False)
    agent = Agent("t", tools=[delete_all], kernel=kernel)
    results = [e.data["result"] for e in agent.stream("go") if e.kind == "tool_result"]
    assert results == ["denied: approval required"] and kernel.client_calls == []


def test_any_tool_sets_up_tool_core(fake):
    kernel = fake(replies=["x"])
    Agent("t", tools=[add], kernel=kernel).run("x")
    assert (SYSCALL_TOOL, "") in kernel.cores


def test_invalid_tool_protocol(fake):
    with pytest.raises(ValueError):
        Agent("t", tool_protocol="xml", kernel=fake())
