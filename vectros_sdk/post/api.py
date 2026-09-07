from typing import Any, Dict, List, Optional, Union

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.post.models import PostQuery, PostResponse


def _parse_post_response(raw_resp: Dict[str, Any]) -> PostResponse:
    """Parse raw response dictionary from kernel into PostResponse object."""
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
    Send a direct message from one agent to another.

    Args:
        sender: Sending agent identifier.
        recipient: Target recipient agent identifier.
        message: Content string or structured data payload.
        metadata: Optional metadata (priority, headers, tags).
        base_url: API endpoint URL.

    Returns:
        PostResponse: Response containing message_id and delivery status.
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
    Fetch pending messages from the agent's mailbox.

    Args:
        agent_name: Receiving agent identifier.
        limit: Maximum number of messages to fetch (default: 10).
        mark_as_read: Whether to acknowledge / mark fetched messages as read.
        base_url: API endpoint URL.

    Returns:
        PostResponse: Response containing the list of received messages.
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
    Broadcast a message to all active agents or to an optional topic.

    Args:
        sender: Sending agent identifier.
        message: Content string or structured data payload.
        topic: Optional topic filter for the broadcast.
        metadata: Optional message metadata.
        base_url: API endpoint URL.

    Returns:
        PostResponse: Response containing broadcast status.
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
    Publish a message to a specific pub/sub channel/topic.

    Args:
        sender: Publishing agent identifier.
        topic: Channel topic name.
        message: Content or payload.
        metadata: Optional metadata.
        base_url: API endpoint URL.

    Returns:
        PostResponse: Response containing publication status.
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
    Subscribe an agent to a pub/sub topic channel.

    Args:
        agent_name: Subscribing agent identifier.
        topic: Topic channel name to subscribe to.
        base_url: API endpoint URL.

    Returns:
        PostResponse: Response containing subscription status.
    """
    query = PostQuery(
        agent_name=agent_name,
        action_type="subscribe",
        topic=topic,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_post_response(raw_response)

