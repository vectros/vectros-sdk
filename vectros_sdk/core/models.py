from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class Query(BaseModel):
    """Base class for all queries sent to AIOS kernel."""
    query_class: str = Field(default="base")
    agent_name: Optional[str] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        data = self.model_dump()
        if item in data:
            return data[item]
        raise KeyError(f"'{type(self).__name__}' object has no key '{item}'")

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default


class Response(BaseModel):
    """Base class for all responses received from AIOS kernel."""
    response_class: str = Field(default="base")
    response_message: Optional[str] = Field(default=None)
    finished: bool = Field(default=False)
    error: Optional[str] = Field(default=None)
    status_code: int = Field(default=200)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __getitem__(self, item: str) -> Any:
        # Support response["response"]["response_message"] nesting as documented in sdk.md examples
        if item == "response":
            return self
        if hasattr(self, item):
            return getattr(self, item)
        data = self.model_dump()
        if item in data:
            return data[item]
        raise KeyError(f"'{type(self).__name__}' object has no key '{item}'")

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default
