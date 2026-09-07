"""
Data models for the Post API (Agent-to-Agent Communication) in Vectros SDK.

Defines `PostQuery` for sending, receiving, broadcasting, and publishing messages,
and `PostResponse` for returning message IDs, delivery confirmations, and mailboxes.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import ConfigDict, Field

from vectros_sdk.core.models import Query, Response


class PostQuery(Query):
    """
    Query model for Agent-to-Agent Communication requests dispatched to the AIOS kernel.

    Attributes:
        query_class (str): Module tag, always 'post'.
        agent_name (Optional[str]): Identifier for the sending/calling agent.
        action_type (Literal['send', 'receive', 'broadcast', 'publish', 'subscribe']): Messaging action.
        recipient (Optional[str]): Recipient agent identifier for direct messages.
        topic (Optional[str]): Channel or topic name for pub/sub and targeted broadcasts.
        message (Optional[Union[str, Dict[str, Any]]]): Message content string or structured payload.
        message_id (Optional[str]): Message identifier for tracking or status queries.
        limit (Optional[int]): Maximum number of messages to fetch during receive operations.
        mark_as_read (bool): Whether fetched messages should be acknowledged as read.
        metadata (Optional[Dict[str, Any]]): Key-value pairs for headers, tags, priority, etc.

    Example:
        >>> from vectros_sdk.post.models import PostQuery
        >>> q = PostQuery(
        ...     agent_name="alice",
        ...     action_type="send",
        ...     recipient="bob",
        ...     message="Hello Bob!"
        ... )
    """
    query_class: str = Field(default="post", description="Module category tag.")
    agent_name: Optional[str] = Field(default=None, description="Sender or subscriber agent identifier.")
    action_type: Literal["send", "receive", "broadcast", "publish", "subscribe"] = Field(
        default="send", description="Messaging operation type."
    )
    recipient: Optional[str] = Field(default=None, description="Target recipient agent name.")
    topic: Optional[str] = Field(default=None, description="Channel/topic name for pub/sub.")
    message: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Message body or payload.")
    message_id: Optional[str] = Field(default=None, description="Unique message ID.")
    limit: Optional[int] = Field(default=None, description="Maximum messages to retrieve.")
    mark_as_read: bool = Field(default=True, description="Mark retrieved messages as read.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Message metadata headers.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PostResponse(Response):
    """
    Response model for Agent-to-Agent Communication operations.

    Attributes:
        response_class (str): Module tag, always 'post'.
        response_message (Optional[str]): Delivery status description or kernel response message.
        message_id (Optional[str]): Assigned message ID for sent messages.
        messages (Optional[List[Dict[str, Any]]]): List of received message objects.
        finished (bool): Whether the messaging operation finished successfully.
        error (Optional[str]): Error message if execution encountered an issue.
        status_code (int): HTTP status code from kernel dispatcher.

    Example:
        >>> from vectros_sdk.post.models import PostResponse
        >>> resp = PostResponse(message_id="msg_001", finished=True)
        >>> print(resp.message_id)
        'msg_001'
    """
    response_class: str = Field(default="post", description="Response category tag.")
    response_message: Optional[str] = Field(default=None, description="Status message from message broker.")
    message_id: Optional[str] = Field(default=None, description="Assigned ID of the transmitted message.")
    messages: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="List of messages retrieved from agent mailbox."
    )
    finished: bool = Field(default=True, description="Operation completion status.")
    error: Optional[str] = Field(default=None, description="Error message if operation failed.")
    status_code: int = Field(default=200, description="HTTP status code from kernel dispatcher.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
