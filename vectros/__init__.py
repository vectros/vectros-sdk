"""Vectros SDK: build AI agents that run on the AIOS kernel.

    from vectros import Agent, tool

    @tool
    def weather(city: str) -> str:
        '''Current weather for a city.'''
        return "31 C, clear"

    agent = Agent("helper", tools=[weather])
    print(agent.run("weather in Pune?"))
"""

from .agent import Agent, Event, Storage, ask_terminal
from .config import allowed_models, choose_model, is_local
from .deep import DeepAgent, SubAgent
from .errors import (KernelError, KernelUnavailable, StepLimitReached, ToolDenied,
                     VectrosError)
from .mcp import MCPError, MCPServer
from .tools import KernelTool, Tool, tool

__version__ = "2.1.1"

__all__ = [
    "Agent", "DeepAgent", "Event", "KernelError", "KernelTool", "KernelUnavailable",
    "MCPError", "MCPServer", "StepLimitReached", "Storage", "SubAgent", "Tool", "ToolDenied",
    "VectrosError", "allowed_models", "ask_terminal", "choose_model", "is_local", "tool",
]
