"""Agent settings from vectros.toml and the environment.

    # vectros.toml
    [agents.reviewer]
    model = "anthropic/claude-opus-5-5"
    max_steps = 6

An Agent named "reviewer" takes these settings. For each setting the first of
these wins: the argument passed to Agent(), the VECTROS_<SETTING> environment
variable (model, max_steps, timeout, tool_protocol and trace only; they apply
to every agent in the process), the agent's table in vectros.toml, the default.

The file is $VECTROS_CONFIG, else the nearest vectros.toml in the current
directory or a parent.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

FILE = "vectros.toml"

# Setting: (accepted types, environment variable or None).
SETTINGS: dict[str, tuple[tuple[type, ...], str | None]] = {
    "model": ((str,), "VECTROS_MODEL"),
    "system": ((str,), None),
    "session": ((str,), None),
    "max_steps": ((int,), "VECTROS_MAX_STEPS"),
    "timeout": ((int, float), "VECTROS_TIMEOUT"),
    "tool_protocol": ((str,), "VECTROS_TOOL_PROTOCOL"),
    "trace": ((bool,), "VECTROS_TRACE"),
}

_TRUE = ("1", "true", "yes", "on")


def config_path() -> Path | None:
    """The vectros.toml in use, or None."""
    explicit = os.environ.get("VECTROS_CONFIG")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ValueError(f"VECTROS_CONFIG={explicit} is not a file")
        return path
    here = Path.cwd()
    for directory in (here, *here.parents):
        if (directory / FILE).is_file():
            return directory / FILE
    return None


def _check(name: str, key: str, value: Any, where: str) -> Any:
    types = SETTINGS[key][0]
    # bool is an int subclass; max_steps = true is a mistake, not 1.
    if not isinstance(value, types) or (bool not in types and isinstance(value, bool)):
        kinds = " or ".join(t.__name__ for t in types)
        raise ValueError(f"{where}: {key} for agent {name!r} must be {kinds}")
    if key in ("max_steps", "timeout") and value <= 0:
        raise ValueError(f"{where}: {key} for agent {name!r} must be above 0")
    if key == "tool_protocol" and value not in ("native", "json"):
        raise ValueError(f'{where}: tool_protocol must be "native" or "json"')
    return value


def _from_env(name: str, key: str, raw: str) -> Any:
    types = SETTINGS[key][0]
    variable = SETTINGS[key][1]
    if bool in types:
        return raw.strip().lower() in _TRUE
    try:
        value = types[-1](raw) if types != (str,) else raw
    except ValueError:
        raise ValueError(f"{variable}={raw!r} is not a number") from None
    return _check(name, key, value, variable)


def agent_settings(name: str) -> dict[str, Any]:
    """Settings for the agent ``name`` from the environment and vectros.toml.

    Only settings that are set appear in the result.
    """
    settings: dict[str, Any] = {}
    path = config_path()
    if path is not None:
        with path.open("rb") as handle:
            try:
                agents = tomllib.load(handle).get("agents", {})
            except tomllib.TOMLDecodeError as exc:
                raise ValueError(f"{path}: {exc}") from None
        if not isinstance(agents, dict):
            raise ValueError(f"{path}: agents must be a table of [agents.<name>] tables")
        table = agents.get(name, {})
        if not isinstance(table, dict):
            raise ValueError(f"{path}: agents.{name} must be a table")
        for key, value in table.items():
            if key not in SETTINGS:
                known = ", ".join(SETTINGS)
                raise ValueError(f"{path}: unknown setting agents.{name}.{key}; use {known}")
            settings[key] = _check(name, key, value, str(path))
    for key, (_, variable) in SETTINGS.items():
        if variable and variable in os.environ:
            settings[key] = _from_env(name, key, os.environ[variable])
    return settings


# --- models ---

ALLOWLIST = "/sys/module/aios/parameters/allowed_llm_models"
PROVIDERS = ("openai", "anthropic", "gemini", "nvidia")


def allowed_models(path: str = ALLOWLIST) -> list[str] | None:
    """The kernel's model allowlist; None when every model is allowed or it
    cannot be read (aios.ko not loaded)."""
    try:
        raw = Path(path).read_text().strip()
    except OSError:
        return None
    models = [item.strip() for item in raw.split(",") if item.strip()]
    return None if models == ["*"] else models


def choose_model(*preferences: str | None, path: str = ALLOWLIST) -> str | None:
    """The first preferred model the kernel allows, else the first allowed one.

    For agents shipped to many machines: name the models you want, best
    first, and run with what this machine's administrator allowed."""
    allowed = allowed_models(path)
    for model in preferences:
        if model and (allowed is None or model in allowed):
            return model
    return allowed[0] if allowed else None


def is_local(model: str | None) -> bool:
    """True when the model runs on the LLM worker's own endpoint (Ollama, vLLM)."""
    prefix, sep, _ = (model or "").partition("/")
    return not (sep and prefix in PROVIDERS)
