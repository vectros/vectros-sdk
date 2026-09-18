"""Tool module for Vectros SDK"""

from vectros_sdk.tool.api import call_tool
from vectros_sdk.tool.approval import ApprovalRefused, call_tool_with_approval
from vectros_sdk.tool.core.base import BaseTool
from vectros_sdk.tool.core.registry import (
    clear_registry,
    get_tool,
    list_registered_tools,
    register_tool,
)
from vectros_sdk.tool.models import ToolQuery, ToolResponse

__all__ = [
    "ToolQuery",
    "ToolResponse",
    "call_tool",
    "call_tool_with_approval",
    "ApprovalRefused",
    "BaseTool",
    "register_tool",
    "get_tool",
    "list_registered_tools",
    "clear_registry",
]
