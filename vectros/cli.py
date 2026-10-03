"""`vectros` command: create, run and deploy agents.

    vectros init helper          # new project in ./helper
    vectros chat                 # talk to the agent locally
    vectros run                  # run the entry script locally
    vectros deploy user@host     # ship to a Vectros OS host as a systemd service
    vectros logs | status | stop
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import runpy
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG = "vectros.toml"
REMOTE_ROOT = ".local/share/vectros/agents"
UNIT = "vectros-agent@.service"

_UNIT_TEXT = """\
[Unit]
Description=Vectros agent %i
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/{root}/%i
ExecStart=%h/{root}/%i/.venv/bin/python -m vectros run
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""

_AGENT_TEMPLATE = '''\
from vectros import Agent, tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


agent = Agent("{name}", tools=[add], system="You are a helpful assistant.")

if __name__ == "__main__":
    print(agent.run("What is 2.5 + 4?"))
'''

_CONFIG_TEMPLATE = '''\
[agent]
name = "{name}"
entry = "agent.py"     # script run by `vectros run` and by the deployed service
object = "agent"       # Agent variable used by `vectros chat`

[deploy]
# host = "user@vectros-host"
'''


class CliError(Exception):
    pass


def load_config(project: Path) -> dict:
    path = project / CONFIG
    if not path.exists():
        raise CliError(f"no {CONFIG} in {project}; run `vectros init` first")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    agent = config.get("agent", {})
    name = agent.get("name", "")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name) or name.startswith("."):
        raise CliError(f"{CONFIG}: agent.name must be 1-64 of A-Z a-z 0-9 . _ -")
    entry = project / agent.get("entry", "agent.py")
    if not entry.resolve().is_relative_to(project.resolve()) or not entry.is_file():
        raise CliError(f"{CONFIG}: entry {agent.get('entry')!r} is not a file in the project")
    return config


def _host(args, config: dict) -> str:
    host = args.host or config.get("deploy", {}).get("host")
    if not host:
        raise CliError("no host: pass user@host or set deploy.host in vectros.toml")
    if host.startswith("-") or not re.fullmatch(r"[A-Za-z0-9._@:\[\]-]+", host):
        raise CliError(f"invalid host {host!r}")
    return host


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, check=True, **kwargs)
    except FileNotFoundError as exc:
        raise CliError(f"{command[0]} is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise CliError(f"{command[0]} failed with exit code {exc.returncode}") from exc


def _ssh(host: str, script: str, **kwargs) -> subprocess.CompletedProcess:
    return _run(["ssh", "--", host, script], **kwargs)


# --- commands ---

def cmd_init(args) -> None:
    project = Path(args.name)
    name = project.name
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name) or name.startswith("."):
        raise CliError("name must be 1-64 of A-Z a-z 0-9 . _ -")
    project.mkdir(parents=True, exist_ok=True)
    for filename, text in ((CONFIG, _CONFIG_TEMPLATE), ("agent.py", _AGENT_TEMPLATE)):
        path = project / filename
        if path.exists():
            raise CliError(f"{path} already exists")
    (project / CONFIG).write_text(_CONFIG_TEMPLATE.format(name=name))
    (project / "agent.py").write_text(_AGENT_TEMPLATE.format(name=name))
    print(f"created {project}/{CONFIG} and {project}/agent.py")
    print(f"next: cd {project} && vectros chat")


def cmd_run(args) -> None:
    project = Path(args.project)
    config = load_config(project)
    entry = (project / config["agent"].get("entry", "agent.py")).resolve()
    os.chdir(project)
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *args.args]
    runpy.run_path(str(entry), run_name="__main__")


def _load_agent(project: Path, config: dict):
    entry = (project / config["agent"].get("entry", "agent.py")).resolve()
    sys.path.insert(0, str(entry.parent))
    spec = importlib.util.spec_from_file_location("vectros_entry", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = config["agent"].get("object", "agent")
    agent = getattr(module, name, None)
    from .agent import Agent
    if not isinstance(agent, Agent):
        raise CliError(f"{entry.name} has no Agent named {name!r} (set agent.object)")
    return agent


def cmd_chat(args) -> None:
    project = Path(args.project)
    agent = _load_agent(project, load_config(project))
    if args.prompt:
        _print_run(agent, " ".join(args.prompt))
        return
    print(f"chatting with {agent.name}; Ctrl-D to quit")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            return
        if prompt:
            _print_run(agent, prompt)


def _print_run(agent, prompt: str) -> None:
    streamed = False
    for event in agent.stream(prompt):
        if event.kind == "token":
            print(event.data, end="", flush=True)
            streamed = True
        elif event.kind == "tool_call":
            print(f"  -> {event.data['tool']} {event.data['args']}", file=sys.stderr)
        elif event.kind == "tool_result":
            print(f"  <- {event.data['result'][:200]}", file=sys.stderr)
        elif event.kind == "answer":
            print() if streamed else print(event.data)


def cmd_deploy(args) -> None:
    project = Path(args.project).resolve()
    config = load_config(project)
    host = _host(args, config)
    name = config["agent"]["name"]
    remote = f"{REMOTE_ROOT}/{name}"
    quoted = shlex.quote(remote)

    print(f"[1/4] checking {host}")
    _ssh(host, "test -e /dev/aios || { echo 'aios.ko is not loaded on this host' >&2; exit 1; }; "
               "python3 -c 'import vectros' 2>/dev/null || "
               "{ echo 'python vectros SDK is not installed on this host' >&2; exit 1; }; "
               f"mkdir -p {quoted}")

    print(f"[2/4] syncing {project} -> {host}:~/{remote}")
    _run(["rsync", "-az", "--delete", "--exclude", ".venv", "--exclude", "__pycache__",
          "--exclude", ".git", "-e", "ssh", "--", f"{project}/", f"{host}:{remote}/"])

    print("[3/4] preparing environment")
    install = (f".venv/bin/pip install -q -r requirements.txt"
               if (project / "requirements.txt").exists() else "true")
    unit = _UNIT_TEXT.format(root=REMOTE_ROOT)
    _ssh(host, f"cd {quoted} && "
               "{ test -x .venv/bin/python || python3 -m venv --system-site-packages .venv; } && "
               f"{install} && mkdir -p ~/.config/systemd/user && "
               f"cat > ~/.config/systemd/user/{UNIT}", input=unit.encode())

    print("[4/4] starting service")
    service = shlex.quote(f"vectros-agent@{name}")
    _ssh(host, f"systemctl --user daemon-reload && systemctl --user enable {service} && "
               f"systemctl --user restart {service} && "
               "{ loginctl show-user \"$USER\" -p Linger | grep -q yes || "
               "echo 'note: run `loginctl enable-linger` on the host so the agent "
               "keeps running after logout' >&2; }")
    print(f"deployed {name} to {host}; see `vectros logs` and `vectros status`")


def _remote_service(args) -> tuple[str, str]:
    config = load_config(Path(args.project))
    return _host(args, config), shlex.quote(f"vectros-agent@{config['agent']['name']}")


def cmd_logs(args) -> None:
    host, service = _remote_service(args)
    follow = " -f" if args.follow else ""
    _ssh(host, f"journalctl --user -u {service} -n {int(args.lines)} --no-pager{follow}")


def cmd_status(args) -> None:
    host, service = _remote_service(args)
    try:
        _ssh(host, f"systemctl --user status {service} --no-pager")
    except CliError:
        pass  # systemctl status exits non-zero for stopped units.


def cmd_stop(args) -> None:
    host, service = _remote_service(args)
    _ssh(host, f"systemctl --user disable --now {service}")
    print("stopped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vectros", description="Build and deploy AIOS agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a new agent project")
    p.add_argument("name")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="run the entry script")
    p.add_argument("-C", "--project", default=".")
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the script")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("chat", help="talk to the agent")
    p.add_argument("-C", "--project", default=".")
    p.add_argument("prompt", nargs="*", help="one-shot prompt; omit for interactive chat")
    p.set_defaults(func=cmd_chat)

    for command, func, text in (("deploy", cmd_deploy, "deploy to a Vectros OS host"),
                                ("logs", cmd_logs, "show the deployed agent's logs"),
                                ("status", cmd_status, "show the deployed agent's status"),
                                ("stop", cmd_stop, "stop and disable the deployed agent")):
        p = sub.add_parser(command, help=text)
        p.add_argument("host", nargs="?", help="user@host (default: deploy.host)")
        p.add_argument("-C", "--project", default=".")
        if command == "logs":
            p.add_argument("-f", "--follow", action="store_true")
            p.add_argument("-n", "--lines", type=int, default=100)
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .errors import VectrosError
    try:
        args.func(args)
    except (CliError, VectrosError) as exc:
        print(f"vectros: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
