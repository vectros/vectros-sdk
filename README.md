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
- **Tracing.** When the administrator enables tracing, runs appear in AIOS
  Trace with kernel queue and lease timings. Tracing is off by default. To
  trace an agent every time it runs, the administrator lists its name in the
  kernel's `trace_agents` parameter; `sudo aiosctl trace enable <agent-id>`
  traces one running agent.
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
    model=None,           # default: the LLM worker's model
    tools=[],             # @tool functions and/or kernel tool names
    system=None,          # system prompt
    session=None,         # keep the conversation across runs and restarts
    max_steps=10,         # maximum model calls per run
    approve=None,         # approve(tool_name, args) -> bool
    timeout=120,          # seconds per model or tool call
    tool_protocol="native",  # or "json" for models without function calling
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
- `@tool` functions run in the agent's own process. Each call is first
  submitted as a kernel client tool call (ABI 4.3). The function runs only
  after the kernel hands the call back, which can be after owner approval.
- When a tool raises an error, the error goes back to the model, and the model
  can try again.
- Tool calls use the model's native function calling. For models without
  function calling, set `tool_protocol="json"`: the agent then asks for JSON
  replies.

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
vectros run                 # run the entry script
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
