"""
Post API (Agent-to-Agent Communication) functional interface for Vectros SDK.

Provides functions for direct point-to-point agent messaging, mailbox polling,
broadcasting announcements, and pub/sub topic messaging.
"""

from typing import Any, Dict, List, Optional, Union

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.post.models import PostQuery, PostResponse


def _parse_post_response(raw_resp: Dict[str, Any]) -> PostResponse:
    """
    Parse raw response dictionary from kernel into a typed PostResponse object.

    Args:
        raw_resp (Dict[str, Any]): Raw JSON response from AIOS kernel.

    Returns:
        PostResponse: Typed PostResponse instance with unwrapped fields.
    """
    if isinstance(raw_resp, dict):
        if "response" in raw_resp and isinstance(raw_resp["response"], dict):
            inner = dict(raw_resp["response"])
            if "status_code" not in inner and "status_code" in raw_resp:
                inner["status_code"] = raw_resp["status_code"]
            if "error" not in inner and "error" in raw_resp:
                inner["error"] = raw_resp["error"]
            if "finished" not in inner and "finished" in raw_resp:
                inner["finished"] = raw_resp["finished"]
            return PostResponse(**inner)
        return PostResponse(**raw_resp)
    return PostResponse(response_message=str(raw_resp))


def send_post(
    sender: str,
    recipient: str,
    message: Union[str, Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    base_url: str = aios_kernel_url,
) -> PostResponse:
    """
    Send a direct message from one agent to another agent's mailbox.

    Args:
        sender (str): Sending agent identifier.
        recipient (str): Target recipient agent identifier.
        message (Union[str, Dict[str, Any]]): Text message content or structured payload.
        metadata (Optional[Dict[str, Any]], optional): Optional metadata (headers, tags, priority). Defaults to None.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        PostResponse: Response object containing `message_id` and delivery confirmation.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.post.api import send_post
        >>> resp = send_post(
        ...     sender="alice",
        ...     recipient="bob",
        ...     message="Please analyze dataset #42",
        ...     metadata={"priority": "high"}
        ... )
        >>> print(resp.message_id)
    """
    query = PostQuery(
        agent_name=sender,
        action_type="send",
        recipient=recipient,
        message=message,
        metadata=metadata,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_post_response(raw_response)


def receive_posts(
    agent_name: str,
    limit: int = 10,
    mark_as_read: bool = True,
    base_url: str = aios_kernel_url,
) -> PostResponse:
    """
    Fetch pending messages from an agent's inbox/mailbox.

    Args:
        agent_name (str): Receiving agent identifier.
        limit (int, optional): Maximum number of messages to retrieve. Defaults to 10.
        mark_as_read (bool, optional): Whether retrieved messages should be marked as read. Defaults to True.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        PostResponse: Response object containing `messages` list with message IDs, senders, and content.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.post.api import receive_posts
        >>> resp = receive_posts(agent_name="bob", limit=5)
        >>> for msg in resp.messages:
        ...     print(msg["sender"], msg["content"])
    """
    query = PostQuery(
        agent_name=agent_name,
        action_type="receive",
        limit=limit,
        mark_as_read=mark_as_read,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_post_response(raw_response)


def broadcast_post(
    sender: str,
    message: Union[str, Dict[str, Any]],
    topic: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    base_url: str = aios_kernel_url,
) -> PostResponse:
    """
    Broadcast a message to all active agents in the system or to an optional topic channel.

    Args:
        sender (str): Sending agent identifier.
        message (Union[str, Dict[str, Any]]): Broadcast text message or payload.
        topic (Optional[str], optional): Target topic name for filtered broadcasts. Defaults to None.
        metadata (Optional[Dict[str, Any]], optional): Metadata headers. Defaults to None.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        PostResponse: Response object containing broadcast delivery status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.post.api import broadcast_post
        >>> resp = broadcast_post(sender="admin_bot", message="System reboot at 2:00 AM UTC")
        >>> print(resp.response_message)
    """
    query = PostQuery(
        agent_name=sender,
        action_type="broadcast",
        topic=topic,
        message=message,
        metadata=metadata,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_post_response(raw_response)


def publish_to_topic(
    sender: str,
    topic: str,
    message: Union[str, Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    base_url: str = aios_kernel_url,
) -> PostResponse:
    """
    Publish a message to a specific pub/sub topic channel.

    Args:
        sender (str): Publishing agent identifier.
        topic (str): Channel topic name to publish to.
        message (Union[str, Dict[str, Any]]): Payload or message content.
        metadata (Optional[Dict[str, Any]], optional): Additional metadata. Defaults to None.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        PostResponse: Response object containing publication confirmation.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.post.api import publish_to_topic
        >>> resp = publish_to_topic(
        ...     sender="sensor_agent",
        ...     topic="telemetry/temperature",
        ...     message={"celsius": 23.4}
        ... )
    """
    query = PostQuery(
        agent_name=sender,
        action_type="publish",
        topic=topic,
        message=message,
        metadata=metadata,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_post_response(raw_response)


def subscribe_topic(
    agent_name: str,
    topic: str,
    base_url: str = aios_kernel_url,
) -> PostResponse:
    """
    Subscribe an agent to a pub/sub topic channel to receive published messages.

    Args:
        agent_name (str): Subscribing agent identifier.
        topic (str): Topic channel name to subscribe to.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        PostResponse: Response object containing subscription status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.post.api import subscribe_topic
        >>> resp = subscribe_topic(agent_name="alert_bot", topic="telemetry/temperature")
        >>> print(resp.response_message)
    """
    query = PostQuery(
        agent_name=agent_name,
        action_type="subscribe",
        topic=topic,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_post_response(raw_response)
