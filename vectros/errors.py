"""Exceptions raised by the Vectros SDK."""


class VectrosError(Exception):
    """Base class for every SDK error."""


class KernelUnavailable(VectrosError):
    """libaios.so or /dev/aios is missing; Vectros agents need aios.ko."""


class KernelError(VectrosError):
    """A syscall failed in the kernel or in the worker that served it."""

    def __init__(self, errno: int, message: str):
        super().__init__(message)
        self.errno = errno


class ToolDenied(VectrosError):
    """The owner or the kernel refused a tool call."""

    def __init__(self, tool: str, reason: str = ""):
        super().__init__(f"tool {tool!r} denied" + (f": {reason}" if reason else ""))
        self.tool = tool
        self.reason = reason


class StepLimitReached(VectrosError):
    """The agent used max_steps without producing a final answer."""
