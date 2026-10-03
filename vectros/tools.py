"""Turn plain Python functions into agent tools.

    @tool
    def weather(city: str, units: str = "metric") -> str:
        '''Current weather for a city.

        Args:
            city: City name, e.g. "Pune".
        '''
"""

from __future__ import annotations

import inspect
import json
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

_JSON_TYPES: dict[Any, str] = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", tuple: "array", dict: "object",
}


def _json_schema(annotation: Any) -> dict:
    """JSON Schema for a type hint; unknown types become an empty schema."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is Literal:
        return {"enum": list(args)}
    if origin in (typing.Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        return _json_schema(options[0]) if len(options) == 1 else {}
    if origin in (list, tuple, set, frozenset):
        schema: dict = {"type": "array"}
        if args and args[0] is not Ellipsis:
            schema["items"] = _json_schema(args[0])
        return schema
    if origin is dict:
        return {"type": "object"}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    return {}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Summary paragraph plus Google-style ``Args:`` descriptions."""
    doc = inspect.cleandoc(doc or "")
    summary, _, rest = doc.partition("\n\n")
    params: dict[str, str] = {}
    match = re.search(r"^Args:\s*\n((?:[ \t]+.*\n?)+)", doc, re.M)
    if match:
        for line in match.group(1).splitlines():
            item = re.match(r"\s+(\w+)(?:\s*\([^)]*\))?:\s*(.+)", line)
            if item:
                params[item.group(1)] = item.group(2).strip()
    if summary.startswith("Args:"):
        summary = ""
    return " ".join(summary.split()), params


@dataclass
class Tool:
    """A function the agent may call. Runs in the agent's own process."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    approval: bool = False

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def invoke(self, args: dict) -> str:
        allowed = set(self.parameters.get("properties", {}))
        unknown = set(args) - allowed
        if unknown:
            raise TypeError(f"unexpected arguments: {', '.join(sorted(unknown))}")
        result = self.fn(**args)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


@dataclass
class KernelTool:
    """A tool registered in the AIOS kernel (admin tools, MCP servers)."""

    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    approval: bool = False


def tool(fn: Callable | None = None, *, name: str | None = None,
         description: str | None = None, approval: bool = False):
    """Decorator that makes a function an agent tool.

    Use bare (``@tool``) or with options (``@tool(approval=True)``). Tools
    marked ``approval=True`` run only after the agent's approver allows them.
    """

    def wrap(func: Callable) -> Tool:
        summary, arg_docs = _parse_docstring(func.__doc__ or "")
        hints = typing.get_type_hints(func)
        properties, required = {}, []
        for param in inspect.signature(func).parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            schema = _json_schema(hints.get(param.name, param.annotation))
            if param.name in arg_docs:
                schema["description"] = arg_docs[param.name]
            if param.default is param.empty:
                required.append(param.name)
            else:
                schema["default"] = param.default
            properties[param.name] = schema
        parameters: dict = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return Tool(name or func.__name__, description or summary or func.__name__,
                    parameters, func, approval)

    return wrap(fn) if fn is not None else wrap
