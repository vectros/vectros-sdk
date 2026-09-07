"""
Core data models for the Vectros SDK.

Defines the fundamental `Query` and `Response` base classes inherited by all
subsystem modules (LLM, Memory, Storage, Tool, and Post) to communicate with
the AIOS kernel over HTTP.
"""

from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class Query(BaseModel):
    """
    Base class for all query requests dispatched to the AIOS kernel.

    Attributes:
        query_class (str): Classification type for the query (e.g. 'llm', 'memory',
            'storage', 'tool', 'post').
        agent_name (Optional[str]): Namespace identifier for the agent originating
            the request.

    Example:
        >>> from vectros_sdk.core.models import Query
        >>> q = Query(query_class="custom", agent_name="assistant_1")
        >>> print(q["agent_name"])
        'assistant_1'
    """
    query_class: str = Field(default="base", description="Query classification tag for routing in the kernel.")
    agent_name: Optional[str] = Field(default=None, description="Identifier for the agent sending the query.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __getitem__(self, item: str) -> Any:
        """
        Enable dictionary-style subscript access to model attributes and extra fields.

        Args:
            item (str): Field name or attribute key to access.

        Returns:
            Any: Value of the requested field.

        Raises:
            KeyError: If the field does not exist on the query object.
        """
        if hasattr(self, item):
            return getattr(self, item)
        data = self.model_dump()
        if item in data:
            return data[item]
        raise KeyError(f"'{type(self).__name__}' object has no key '{item}'")

    def get(self, item: str, default: Any = None) -> Any:
        """
        Safely retrieve an attribute value with a fallback default.

        Args:
            item (str): Field name or attribute key to retrieve.
            default (Any, optional): Fallback value if key is not found. Defaults to None.

        Returns:
            Any: Value of the field if present, otherwise default.
        """
        try:
            return self[item]
        except KeyError:
            return default


class Response(BaseModel):
    """
    Base class for all responses received from the AIOS kernel.

    Supports dual-access indexing: both flat (`response['response_message']`) and
    nested (`response['response']['response_message']`) as documented in SDK guides.

    Attributes:
        response_class (str): Module classification of the response (e.g. 'llm', 'memory').
        response_message (Optional[str]): Human-readable message or serialized payload.
        finished (bool): Whether the requested operation has finished executing.
        error (Optional[str]): Error description if operation failed, otherwise None.
        status_code (int): HTTP status code returned by the kernel (defaults to 200).

    Example:
        >>> from vectros_sdk.core.models import Response
        >>> r = Response(response_message="Success", finished=True)
        >>> print(r["response"]["response_message"])
        'Success'
    """
    response_class: str = Field(default="base", description="Module category tag of the response.")
    response_message: Optional[str] = Field(default=None, description="Result message or text content.")
    finished: bool = Field(default=False, description="Indicates if the operation completed successfully.")
    error: Optional[str] = Field(default=None, description="Error message if the operation failed.")
    status_code: int = Field(default=200, description="HTTP status code from kernel dispatcher.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __getitem__(self, item: str) -> Any:
        """
        Enable dictionary-style subscript access with support for nested ['response'] access.

        Args:
            item (str): Attribute name or 'response' for self-referential unwrap.

        Returns:
            Any: Attribute value or self if item == 'response'.

        Raises:
            KeyError: If the key does not exist on the response object.
        """
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
        """
        Safely retrieve an attribute value with a fallback default.

        Args:
            item (str): Attribute key to retrieve.
            default (Any, optional): Fallback value if key is not found. Defaults to None.

        Returns:
            Any: Value of the attribute if found, otherwise default.
        """
        try:
            return self[item]
        except KeyError:
            return default
