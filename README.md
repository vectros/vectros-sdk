# Vectros SDK

Build AI agents that run on the AIOS kernel (`aios.ko`) in a few lines, then
deploy them to a Vectros OS host.

```python
from vectros import Agent, tool

@tool
def weather(city: str) -> str:
    """Current weather for a city."""
    return f"{city}: 31 C, clear sky"

agent = Agent("helper", tools=[weather])
print(agent.run("What is the weather in Pune?"))
```

The SDK registers the agent with the kernel, sets up its cores, runs the tool
loop, streams output, emits traces and unregisters the agent at exit.

## What the kernel gives every agent

- **Scheduling and quotas.** Every model call is an AIOS LLM syscall. The kernel
  queues it fairly with other agents and enforces per-agent limits.
- **Model allowlist.** `model=` must be on the administrator's allowlist.
- **Tool control.** Every tool call is a kernel syscall, including your own
  `@tool` functions. The kernel checks permission, applies the deadline and
  records the call.
- **Approvals.** Kernel tools flagged as side-effecting, and `@tool(approval=True)`
  functions, are held until the owner approves them, through `approve=` or in
  AIOS Manager.
- **Stable identity.** The agent ID comes from your UID and the agent name.
  Only one agent with that name runs per user, and its storage survives
  restarts.
- **Tracing.** Runs appear in AIOS Trace with kernel queue and lease
  timings. Tracing is off by default. Turn it on for your own agent with
  `Agent(..., trace=True)`, `VECTROS_TRACE=1` or `vectros run --trace`
  (kernel ABI 4.5 or later). The administrator can also trace agents by name
  with the kernel's `trace_agents` parameter, or one running agent with
  `sudo aiosctl trace enable <agent-id>`. An agent cannot turn off tracing
  that the administrator turned on.
- **Storage.** `agent.storage` is private, versioned and quota-limited.

The SDK needs `aios.ko` loaded and `libaios.so` installed. There is no
userspace fallback. Without the kernel, the SDK raises `KernelUnavailable`.

## Requirements

- Python 3.11 or later
- `aios.ko` loaded, with the LLM worker running. Storage and kernel tools also
  need the storage and tool workers.
- Your user in the `aios` group
- `libaios.so` in `/usr/lib`, or its path in `VECTROS_LIBAIOS`

```sh
pip install vectros-sdk
```

The package installs as `vectros-sdk` and imports as `vectros`. On Vectros OS
it is preinstalled as the `python-vectros` package.

## Agent

```python
Agent(
    name,                 # shown in AIOS Manager and Trace
    model=None,           # default: the LLM worker's model; or "anthropic/..." etc.
    tools=[],             # @tool functions, kernel tool names, MCPServers
    system=None,          # system prompt
    session=None,         # keep the conversation across runs and restarts
    max_steps=10,         # maximum model calls per run
    approve=None,         # approve(tool_name, args) -> bool
    timeout=120,          # seconds per model or tool call
    tool_protocol="native",  # or "json" for models without function calling
    trace=None,           # True: trace this agent's runs (default: VECTROS_TRACE)
)
```

| Call | Result |
| --- | --- |
| `agent.run(prompt)` | Final answer as `str` |
| `agent.stream(prompt)` | `Event`s: `token`, `tool_call`, `tool_result`, `answer` |
| `agent.storage.write(path, data)` / `.read(path)` | Kernel storage |
| `agent.reset()` | Forget the session |
| `agent.close()` or `with Agent(...)` | Unregister now, not at exit |

```python
for event in agent.stream("Plan my day"):
    if event.kind == "token":
        print(event.data, end="")
```

### Settings outside the code

Settings you do not pass to `Agent()` come from the environment, then from
the agent's table in `vectros.toml`, then from the defaults above. Change the
model or the limits without editing code:

```toml
[agents.reviewer]                 # applies to Agent("reviewer")
model = "anthropic/claude-opus-5-5"
system = "You are a reviewer."
session = "reviewer-demo"
max_steps = 6
timeout = 90
tool_protocol = "native"
trace = true
```

| Environment variable | Setting (every agent in the process) |
| --- | --- |
| `VECTROS_MODEL` | `model` |
| `VECTROS_MAX_STEPS` | `max_steps` |
| `VECTROS_TIMEOUT` | `timeout` |
| `VECTROS_TOOL_PROTOCOL` | `tool_protocol` |
| `VECTROS_TRACE` | `trace` |
| `VECTROS_CONFIG` | Path of the `vectros.toml` to use |

Without `VECTROS_CONFIG`, the SDK reads the nearest `vectros.toml` in the
current directory or a parent. An unknown setting or a wrong type raises
`ValueError` when the agent is created. `tools` and `approve` are code, so
they stay in `Agent()`. `DeepAgent` reads the same table; its `max_steps`
default is 50.

## Models

Every model call goes through the kernel, so the LLM worker decides where it
runs. A plain name (`gemma4:e4b`) goes to the worker's local endpoint (Ollama
or vLLM). A `<provider>/<model>` name goes to a hosted provider when the
administrator has set that provider's key for the worker in
`/etc/aios/aios.env`:

| Provider | `model=` example | Worker key |
| --- | --- | --- |
| OpenAI | `openai/gpt-5` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| NVIDIA NIM | `nvidia/meta/llama-3.3-70b-instruct` | `NVIDIA_API_KEY` |

```python
agent = Agent("helper", model="anthropic/claude-opus-5-5", tools=[weather])
```

The model must also be on the kernel's allowlist (`allowed_llm_models`).
`vectros models` lists the allowlist and where each model runs. API keys stay
with the worker; agents never see them. Without the key, the name goes to the
local endpoint unchanged, so a vLLM model called `openai/gpt-oss-20b` still
works.

## Tools

```python
@tool
def search(query: str, limit: int = 5) -> list[str]:
    """Search the docs.

    Args:
        query: What to look for.
    """

@tool(approval=True)          # runs only if approve() returns True
def send_email(to: str, body: str) -> str: ...

agent = Agent("ops", tools=[search, send_email, "search_web"], approve=ask_terminal)
```

- The SDK builds the parameter schema from type hints, and descriptions from
  the docstring and its `Args:` section.
- A string names a tool registered in the kernel, for example an admin tool or
  an MCP tool such as `github__create_issue`. These run in the AIOS tool worker.
  A name ending in `?` (`"fetch__fetch?"`) is optional: the agent gets it when
  the tool worker registered it and runs without it otherwise.
- `@tool` functions run in the agent's own process. Each call is first
  submitted as a kernel client tool call (ABI 4.3). The function runs only
  after the kernel hands the call back, which can be after owner approval.
- When a tool raises an error, the error goes back to the model, and the model
  can try again.
- Tool calls use the model's native function calling. For models without
  function calling, set `tool_protocol="json"`: the agent then asks for JSON
  replies.

## MCP servers

```python
from vectros import Agent, MCPServer

files = MCPServer("files", "npx -y @modelcontextprotocol/server-filesystem .")
github = MCPServer("github", url="https://api.githubcopilot.com/mcp/",
                   headers={"Authorization": f"Bearer {token}"})
agent = Agent("ops", tools=[files, github], approve=ask_terminal)
```

- The agent's process starts the server (stdio) or connects to it (Streamable
  HTTP). The SDK has its own MCP client with no extra dependencies.
- Each MCP tool becomes `<server>__<tool>` and runs as a kernel client tool
  call, like a `@tool` function: the kernel checks permission, applies the
  deadline, records the call and holds it for approval when needed.
- Tools need approval unless the server marks them read-only
  (`readOnlyHint`). Override with `approval=False`, `approval=True` or a list
  of tool names. `tools=[...]` offers only the named server tools.
- `MCPServer.from_config("mcp.json")` reads the `{"mcpServers": {...}}` format
  of other MCP hosts and the AIOS tool worker. `${VAR}` reads the environment.
- `server.close()` or `with MCPServer(...)` stops the server; otherwise it
  stops at exit.

For servers that every agent on a host shares, the administrator registers
them in the AIOS tool worker instead, and agents name the tools by string.

## DeepAgent

A `DeepAgent` handles long, multi-step work. It plans, keeps notes in files
and hands parts of the work to subagents.

```python
from vectros import DeepAgent, SubAgent

researcher = SubAgent("researcher", "Researches one question in depth.",
                      system="Research thoroughly. Cite sources.",
                      tools=["search_web"], model="openai/gpt-5")
agent = DeepAgent("analyst", tools=[search], subagents=[researcher],
                  model="anthropic/claude-opus-5-5", session="q3")
print(agent.run("Compare the three largest Linux schedulers"))
print(agent.todos, agent.files)
```

It takes every `Agent` argument (default `max_steps=50`) and adds these tools:

| Tool | Purpose |
| --- | --- |
| `write_todos` | Keep a plan of pending, in-progress and completed steps |
| `ls`, `read_file`, `write_file`, `edit_file` | A private file space for notes and drafts |
| `task` | Hand a self-contained job to a subagent |

- A subagent is a separate kernel agent, `<parent>.<name>`, with its own
  scheduling, quotas and trace. It starts with a clean context, shares the
  parent's file space and returns only its final answer.
- A `general-purpose` subagent with the parent's tools is always available.
- With `session=`, the plan and files are saved next to the session history.

## Sessions

```python
agent = Agent("support", session="customer-42")
```

History is stored in `$XDG_STATE_HOME/vectros/sessions/<agent>/<session>.json`
and survives restarts. Only user prompts and final answers are kept, up to 40
messages.

## CLI

```sh
vectros init helper         # helper/vectros.toml + helper/agent.py
cd helper
vectros chat                # interactive; or: vectros chat "one question"
vectros run                 # run the entry script (--trace to trace it)
vectros models              # models agents may request, and where they run
vectros deploy me@host      # deploy to a Vectros OS host
vectros logs -f             # follow the deployed agent's journal
vectros status
vectros stop
```

`vectros.toml`:

```toml
[agent]
name = "helper"
entry = "agent.py"     # script run by `vectros run` and by the service
object = "agent"       # Agent variable used by `vectros chat`

[deploy]
host = "me@vectros-host"
```

The host needs SSH and rsync. VectrOS ships both with sshd off; the owner
turns it on with `sudo systemctl enable --now sshd`. Use key authentication.

`vectros deploy` does the following:

1. Checks that the host has `/dev/aios` and the `vectros` package.
2. Copies the project with rsync to `~/.local/share/vectros/agents/<name>`.
3. Creates a venv with system site packages and installs `requirements.txt`
   if the project has one.
4. Starts the systemd user service `vectros-agent@<name>`, which runs
   `python -m vectros run`.

To keep the agent running after you log out, run `loginctl enable-linger` on
the host.

## Tests

```sh
pytest                         # unit tests, fake kernel
VECTROS_E2E=1 pytest -m kernel # real aios.ko and workers
```

## Kernel versions

The SDK needs ABI 4. Kernels before 4.3 still work with these limits:

- Agent IDs are random, so `agent.storage` data is not readable after a restart.
- `@tool` functions run in-process without kernel mediation. The SDK enforces
  `approval=True` itself and denies the call when no `approve=` is set.

## Limits

- The kernel sees that a `@tool` function ran, but it cannot see what the
  function does inside the agent process.
- Sessions are local files on the host, not kernel storage. Kernel storage
  files are limited to 64 KiB.
- Native tool calls need the LLM worker that returns the tool envelope
  (vectros-kernel with ABI 4.3). An older worker returns plain text, so the
  agent treats the reply as the final answer.
