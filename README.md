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
- **Approvals.** Kernel tools flagged as side-effecting are held until the owner
  approves them, through `approve=` or in AIOS Manager.
- **Tracing.** Runs appear in AIOS Trace with kernel queue and lease timings.
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
pip install .
```

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
  an MCP tool such as `github__create_issue`. These run in the AIOS tool worker,
  under kernel permissions and approvals.
- `@tool` functions run in the agent's own process. When a tool raises an
  error, the error goes back to the model, and the model can try again.

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

## Limits in this version

- `@tool` functions run in the agent process, so the kernel does not see or
  audit them. A kernel permit syscall for these calls is planned.
- The kernel assigns a new agent ID on each registration. Kernel storage is
  keyed by agent ID, so `agent.storage` data is not readable after a restart.
  Sessions use local files for this reason.
- The LLM worker returns text only, so tool calls use a JSON reply protocol,
  not native function calling.
