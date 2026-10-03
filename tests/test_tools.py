from typing import Literal, Optional

import pytest

from vectros import Tool, tool


def test_bare_decorator_builds_schema_from_hints_and_docstring():
    @tool
    def weather(city: str, days: int = 1, units: Literal["c", "f"] = "c") -> str:
        """Current weather for a city.

        Args:
            city: City name.
            days (int): How many days.
        """
        return city

    assert isinstance(weather, Tool)
    assert weather.name == "weather"
    assert weather.description == "Current weather for a city."
    assert weather.parameters == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name."},
            "days": {"type": "integer", "description": "How many days.", "default": 1},
            "units": {"enum": ["c", "f"], "default": "c"},
        },
        "required": ["city"],
    }
    assert weather("Pune") == "Pune"


def test_options_and_container_types():
    @tool(name="lookup", approval=True)
    def find(ids: list[int], note: Optional[str] = None, meta: dict = None):
        return {"ids": ids}

    assert find.name == "lookup"
    assert find.approval is True
    assert find.description == "find"
    assert find.parameters["properties"]["ids"] == {"type": "array", "items": {"type": "integer"}}
    assert find.parameters["properties"]["note"] == {"type": "string", "default": None}
    assert find.parameters["properties"]["meta"]["type"] == "object"


def test_invoke_serializes_and_rejects_unknown_args():
    @tool
    def pair(a: int, b: int):
        return [a, b]

    assert pair.invoke({"a": 1, "b": 2}) == "[1, 2]"
    with pytest.raises(TypeError, match="unexpected arguments: c"):
        pair.invoke({"a": 1, "b": 2, "c": 3})
