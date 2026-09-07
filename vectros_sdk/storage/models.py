"""
Data models for the Storage API in Vectros SDK.

Defines `StorageQuery` for file and directory filesystem requests and
`StorageResponse` for returning operation status, file data, and search matches.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import ConfigDict, Field

from vectros_sdk.core.models import Query, Response


class StorageQuery(Query):
    """
    Query model for Storage API requests dispatched to the AIOS kernel.

    Attributes:
        query_class (str): Module classification tag, always 'storage'.
        agent_name (Optional[str]): Namespace identifier for the agent.
        operation_type (str): Storage operation name (e.g. 'mount', 'create_file',
            'create_dir', 'write_file', 'retrieve_file', 'rollback_file', 'share_file').
        params (Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]): List or dictionary
            of operation-specific parameters.

    Example:
        >>> from vectros_sdk.storage.models import StorageQuery
        >>> q = StorageQuery(
        ...     agent_name="dev_agent",
        ...     operation_type="create_file",
        ...     params=[{"file_path": "main.py"}]
        ... )
    """
    query_class: str = Field(default="storage", description="Module category tag.")
    agent_name: Optional[str] = Field(default=None, description="Identifier for the agent.")
    operation_type: str = Field(default="text", description="Filesystem operation to execute.")
    params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = Field(
        default=None, description="Parameters dictionary or list for the operation."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class StorageResponse(Response):
    """
    Response model for Storage API filesystem operations.

    Attributes:
        response_class (str): Module tag, always 'storage'.
        response_message (Optional[str]): Status message or file operation output.
        finished (bool): Whether the storage operation succeeded.
        error (Optional[str]): Error message if execution encountered a failure.
        status_code (int): HTTP response code from kernel dispatcher.

    Example:
        >>> from vectros_sdk.storage.models import StorageResponse
        >>> resp = StorageResponse(response_message="File written", finished=True)
        >>> print(resp["response"]["response_message"])
        'File written'
    """
    response_class: str = Field(default="storage", description="Response category tag.")
    response_message: Optional[str] = Field(default=None, description="Result message or data from kernel.")
    finished: bool = Field(default=False, description="Whether the operation completed successfully.")
    error: Optional[str] = Field(default=None, description="Error message if operation failed.")
    status_code: int = Field(default=200, description="HTTP response status code.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
