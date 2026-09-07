"""
Storage API functional interface for Vectros SDK.

Provides functions for directory mounting, file creation, writing, semantic retrieval,
version rollback, and inter-agent file sharing within the AIOS kernel.
"""

from typing import Any, Dict, List, Optional

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.storage.models import StorageQuery, StorageResponse


def _parse_storage_response(raw_resp: Dict[str, Any]) -> StorageResponse:
    """
    Parse raw response dictionary from kernel into a typed StorageResponse object.

    Args:
        raw_resp (Dict[str, Any]): Raw JSON response from AIOS kernel.

    Returns:
        StorageResponse: Typed StorageResponse instance with unwrapped fields.
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
            return StorageResponse(**inner)
        return StorageResponse(**raw_resp)
    return StorageResponse(response_message=str(raw_resp))


def mount(
    agent_name: str,
    root_dir: str,
    base_url: str = aios_kernel_url,
) -> StorageResponse:
    """
    Mount a directory as the root storage location for an agent.

    Args:
        agent_name (str): Identifier for the agent making the request.
        root_dir (str): Directory path on the device/kernel to mount as root.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        StorageResponse: Response object containing operation status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.storage.api import mount
        >>> resp = mount("research_agent", "/data/research_projects")
        >>> print(resp.response_message)
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
    Create a new empty file at the specified relative path.

    Args:
        agent_name (str): Identifier for the agent making the request.
        file_path (str): Relative path where the new file should be created.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        StorageResponse: Response object containing file creation status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.storage.api import create_file
        >>> resp = create_file("developer_agent", "projects/main.py")
        >>> print(resp.finished)
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
    Create a new directory structure at the specified relative path.

    Args:
        agent_name (str): Identifier for the agent making the request.
        dir_path (str): Directory path to create.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        StorageResponse: Response object containing directory creation status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.storage.api import create_dir
        >>> resp = create_dir("project_manager", "projects/new_webapp/src")
        >>> print(resp.response_message)
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
    Write text content to a file, creating parent directories and the file if they do not exist.

    Args:
        agent_name (str): Identifier for the agent making the request.
        file_path (str): Relative path of the file to write.
        content (str): Text content to write into the file.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        StorageResponse: Response object containing write operation status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.storage.api import write_file
        >>> resp = write_file("web_developer", "index.html", "<h1>Hello AIOS</h1>")
        >>> print(resp.response_message)
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
    Search and retrieve files matching natural language query criteria and keywords.

    Args:
        agent_name (str): Identifier for the agent making the request.
        query_text (str): Search description or query phrase.
        n (int): Maximum number of matching files to return.
        keywords (Optional[List[str]], optional): Specific keywords to match. Defaults to None.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        StorageResponse: Response object containing matching file paths and information.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.storage.api import retrieve_file
        >>> resp = retrieve_file("data_scientist", "data processing pipeline", n=5, keywords=["pandas"])
        >>> print(resp.response_message)
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
    Revert a file to an earlier version in its version history.

    Args:
        agent_name (str): Identifier for the agent making the request.
        file_path (str): Path of the file to revert.
        n (int): Number of versions to roll back (1 for previous version).
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        StorageResponse: Response object containing rollback status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.storage.api import rollback_file
        >>> resp = rollback_file("system_agent", "config/app.json", n=1)
        >>> print(resp.response_message)
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
    Share a file across the AIOS kernel environment to make it accessible to other agents.

    Args:
        agent_name (str): Identifier for the agent sharing the file.
        file_path (str): Path of the file to share.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        StorageResponse: Response object containing sharing status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.storage.api import share_file
        >>> resp = share_file("data_engineer", "datasets/processed.csv")
        >>> print(resp.response_message)
    """
    query = StorageQuery(
        agent_name=agent_name,
        operation_type="share_file",
        params=[{"file_path": file_path}],
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_storage_response(raw_response)
