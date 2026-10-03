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
from .errors import (KernelError, KernelUnavailable, StepLimitReached, ToolDenied,
                     VectrosError)
from .tools import KernelTool, Tool, tool

__version__ = "2.0.0"

__all__ = [
    "Agent", "Event", "KernelError", "KernelTool", "KernelUnavailable", "StepLimitReached",
    "Storage", "Tool", "ToolDenied", "VectrosError", "ask_terminal", "tool",
]
