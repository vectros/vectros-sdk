from typing import Any, Dict, List, Optional

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.storage.models import StorageQuery, StorageResponse


def _parse_storage_response(raw_resp: Dict[str, Any]) -> StorageResponse:
    """Parse raw response dictionary from kernel into StorageResponse object."""
    if isinstance(raw_resp, dict):
        if "response" in raw_resp and isinstance(raw_resp["response"], dict):
            inner = dict(raw_resp["response"])
            if "status_code" not in inner and "status_code" in raw_resp:
                inner["status_code"] = raw_resp["status_code"]
            if "error" not in inner and "error" in raw_resp:
                inner["error"] = raw_resp["error"]
            if "finished" not in inner and "finished" in raw_resp:
                inner["finished"] = raw_resp["finished"]
            return StorageResponse(**inner)
        return StorageResponse(**raw_resp)
    return StorageResponse(response_message=str(raw_resp))


def mount(
    agent_name: str,
    root_dir: str,
    base_url: str = aios_kernel_url,
) -> StorageResponse:
    """
    Mounts a directory as the root storage location for an agent.

    Args:
        agent_name: Identifier for the agent making the request.
        root_dir: Directory path to mount as storage root.
        base_url: API endpoint URL (default: configured AIOS kernel URL).

    Returns:
        StorageResponse: Response object containing operation result.
    """
    query = StorageQuery(
        agent_name=agent_name,
        operation_type="mount",
        params=[{"root_dir": root_dir}],
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_storage_response(raw_response)


def create_file(
    agent_name: str,
    file_path: str,
    base_url: str = aios_kernel_url,
) -> StorageResponse:
    """
    Creates a new empty file at the specified path.

    Args:
        agent_name: Identifier for the agent making the request.
        file_path: Path where the new file should be created.
        base_url: API endpoint URL (default: configured AIOS kernel URL).

    Returns:
        StorageResponse: Response object containing creation status.
    """
    query = StorageQuery(
        agent_name=agent_name,
        operation_type="create_file",
        params=[{"file_path": file_path}],
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_storage_response(raw_response)


def create_dir(
    agent_name: str,
    dir_path: str,
    base_url: str = aios_kernel_url,
) -> StorageResponse:
    """
    Creates a new directory at the specified path.

    Args:
        agent_name: Identifier for the agent making the request.
        dir_path: Path where the new directory should be created.
        base_url: API endpoint URL (default: configured AIOS kernel URL).

    Returns:
        StorageResponse: Response object containing creation status.
    """
    query = StorageQuery(
        agent_name=agent_name,
        operation_type="create_dir",
        params=[{"dir_path": dir_path}],
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_storage_response(raw_response)


def write_file(
    agent_name: str,
    file_path: str,
    content: str,
    base_url: str = aios_kernel_url,
) -> StorageResponse:
    """
    Writes content to a file, creating it if it doesn't exist.

    Args:
        agent_name: Identifier for the agent making the request.
        file_path: Path to the file to write.
        content: Text content to write to the file.
        base_url: API endpoint URL (default: configured AIOS kernel URL).

    Returns:
        StorageResponse: Response object containing write operation status.
    """
    query = StorageQuery(
        agent_name=agent_name,
        operation_type="write_file",
        params=[{"file_path": file_path, "content": content}],
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_storage_response(raw_response)


def retrieve_file(
    agent_name: str,
    query_text: str,
    n: int,
    keywords: Optional[List[str]] = None,
    base_url: str = aios_kernel_url,
) -> StorageResponse:
    """
    Searches for files matching query criteria and returns the results.

    Args:
        agent_name: Identifier for the agent making the request.
        query_text: Text to search for in files.
        n: Maximum number of results to return.
        keywords: Optional list of specific keywords to match.
        base_url: API endpoint URL (default: configured AIOS kernel URL).

    Returns:
        StorageResponse: Response object containing matched files.
    """
    param_dict: Dict[str, Any] = {"query_text": query_text, "n": n}
    if keywords is not None:
        param_dict["keywords"] = keywords

    query = StorageQuery(
        agent_name=agent_name,
        operation_type="retrieve_file",
        params=[param_dict],
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_storage_response(raw_response)


def rollback_file(
    agent_name: str,
    file_path: str,
    n: int,
    base_url: str = aios_kernel_url,
) -> StorageResponse:
    """
    Reverts a file to a previous version.

    Args:
        agent_name: Identifier for the agent making the request.
        file_path: Path to the file to roll back.
        n: Number of versions to roll back (1 for previous version).
        base_url: API endpoint URL (default: configured AIOS kernel URL).

    Returns:
        StorageResponse: Response object containing rollback status.
    """
    query = StorageQuery(
        agent_name=agent_name,
        operation_type="rollback_file",
        params=[{"file_path": file_path, "n": n}],
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_storage_response(raw_response)


def share_file(
    agent_name: str,
    file_path: str,
    base_url: str = aios_kernel_url,
) -> StorageResponse:
    """
    Makes a file accessible in the AIOS environment.

    Args:
        agent_name: Identifier for the agent sharing the file.
        file_path: Path to the file to be shared.
        base_url: API endpoint URL (default: configured AIOS kernel URL).

    Returns:
        StorageResponse: Response object containing sharing status.
    """
    query = StorageQuery(
        agent_name=agent_name,
        operation_type="share_file",
        params=[{"file_path": file_path}],
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_storage_response(raw_response)

