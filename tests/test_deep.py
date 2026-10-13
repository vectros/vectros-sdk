import json

import pytest

from vectros import DeepAgent, SubAgent, VectrosError, tool


def call(name, **args):
    return {"aios_llm_result": 1, "content": "", "tool_calls": [
        {"id": f"c_{name}", "function": {"name": name, "arguments": json.dumps(args)}}]}


@tool
def search(q: str) -> str:
    """Search."""
    return f"results for {q}"


def test_builtin_tools_plan_and_files(fake):
    kernel = fake(replies=[
        call("write_todos", todos=[{"content": "research", "status": "in_progress"},
                                   {"content": "write"}]),
        call("write_file", path="notes.md", content="alpha\nbeta\n"),
        call("edit_file", path="notes.md", old_string="beta", new_string="gamma"),
        call("read_file", path="notes.md"),
        call("ls"),
        "final"])
    agent = DeepAgent("deep", tools=[search], system="You are careful.", kernel=kernel)
    results = [e.data["result"] for e in agent.stream("go") if e.kind == "tool_result"]
    assert results[0] == "[~] research\n[ ] write"
    assert agent.files == {"notes.md": "alpha\ngamma\n"}
    assert results[3] == "     1\talpha\n     2\tgamma"
    assert results[4] == "notes.md\t12 chars"
    system = kernel.llm_calls[0]["messages"][0]["content"]
    assert system.startswith("You are careful.") and "write_todos" in system
    names = [spec["function"]["name"] for spec in kernel.llm_calls[0]["tools"]]
    assert names == ["search", *DeepAgent.BUILTIN]


def test_edit_file_needs_a_unique_match(fake):
    kernel = fake(replies=[call("write_file", path="a", content="x x"),
                           call("edit_file", path="a", old_string="x", new_string="y"),
                           "ok"])
    agent = DeepAgent("deep", kernel=kernel)
    results = [e.data["result"] for e in agent.stream("go") if e.kind == "tool_result"]
    assert "occurs 2 times" in results[1]


def test_task_runs_a_subagent_with_clean_context_and_shared_files(fake):
    kernel = fake(replies=[
        call("write_file", path="brief.md", content="facts"),
        call("task", description="summarize brief.md", subagent="writer"),
        # The subagent's own loop:
        call("read_file", path="brief.md"),
        "summary of facts",
        "all done"])
    writer = SubAgent("writer", "Writes summaries.", system="Write tersely.", model="openai/gpt-5")
    agent = DeepAgent("deep", subagents=[writer], model="anthropic/claude-opus-5-5", kernel=kernel)
    events = list(agent.stream("go"))
    results = [e.data["result"] for e in events if e.kind == "tool_result"]
    assert results[1] == "summary of facts"
    assert events[-1].data == "all done"
    assert kernel.registered == ["deep", "deep.writer"]
    sub_first = kernel.llm_calls[2]
    assert sub_first["model"] == "openai/gpt-5"
    assert sub_first["messages"][0]["content"].startswith("Write tersely.")
    assert sub_first["messages"][-1] == {"role": "user", "content": "summarize brief.md"}
    sub_tools = [spec["function"]["name"] for spec in sub_first["tools"]]
    assert "task" not in sub_tools and "read_file" in sub_tools
    agent.close()
    assert len(kernel.unregistered) == 2


def test_subagent_steps_reach_the_callback(fake):
    kernel = fake(replies=[
        call("task", description="look it up", subagent="finder"),
        call("search", q="aios"),
        "found it",
        "done"])
    seen = []
    finder = SubAgent("finder", "Finds things.", tools=[search])
    agent = DeepAgent("deep", subagents=[finder], kernel=kernel,
                      on_subagent_event=lambda name, event: seen.append((name, event.kind)))
    assert agent.run("go") == "done"
    assert ("finder", "tool_call") in seen and ("finder", "tool_result") in seen
    assert seen[-1] == ("finder", "answer")
    agent.close()


def test_unknown_subagent_goes_back_to_model(fake):
    kernel = fake(replies=[call("task", description="x", subagent="nobody"), "ok"])
    agent = DeepAgent("deep", kernel=kernel)
    results = [e.data["result"] for e in agent.stream("go") if e.kind == "tool_result"]
    assert results[0].startswith("error: ValueError: unknown subagent 'nobody'")


def test_reserved_tool_names(fake):
    @tool
    def ls() -> str:
        """Mine."""
        return ""

    agent = DeepAgent("deep", tools=[ls], kernel=fake(replies=["x"]))
    with pytest.raises(VectrosError, match="reserved"):
        agent.run("hi")


def test_plan_and_files_persist_with_the_session(fake):
    kernel = fake(replies=[call("write_file", path="a.md", content="kept"),
                           call("write_todos", todos=[{"content": "s", "status": "completed"}]),
                           "ok"])
    DeepAgent("deep", session="s1", kernel=kernel).run("go")
    again = DeepAgent("deep", session="s1", kernel=fake())
    assert again.files == {"a.md": "kept"}
    assert again.todos == [{"content": "s", "status": "completed"}]
    again.reset()
    assert DeepAgent("deep", session="s1", kernel=fake()).files == {}
