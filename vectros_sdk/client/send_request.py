import json
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel
import requests

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.core.models import Query


class AIOSKernelError(Exception):
    """Exception raised when kernel communication or execution fails."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


def send_request(
    query: Union[Query, BaseModel, Dict[str, Any]],
    base_url: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Central dispatcher function to communicate with AIOS kernel via HTTP requests.

    Args:
        query: Query object, Pydantic model, or dictionary payload.
        base_url: AIOS kernel base endpoint URL. Defaults to configured aios_kernel_url.
        timeout: Request timeout in seconds (default 60).

    Returns:
        Dict[str, Any]: Parsed response dictionary from the kernel.

    Raises:
        AIOSKernelError: If request fails or kernel returns non-2xx status code.
    """
    url = (base_url or aios_kernel_url).rstrip("/")
    if not url.endswith("/query"):
        endpoint = f"{url}/query"
    else:
        endpoint = url

    if isinstance(query, BaseModel):
        payload = query.model_dump()
    elif isinstance(query, dict):
        payload = query
    else:
        raise ValueError(f"Expected Query, BaseModel, or dict, got {type(query).__name__}")

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise AIOSKernelError(f"Request to AIOS kernel timed out: {exc}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise AIOSKernelError(f"Failed to connect to AIOS kernel at {endpoint}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise AIOSKernelError(f"Error communicating with AIOS kernel: {exc}") from exc

    if not response.ok:
        error_msg = f"AIOS kernel returned status code {response.status_code}: {response.text}"
        try:
            res_json = response.json()
        except Exception:
            res_json = None
        raise AIOSKernelError(error_msg, status_code=response.status_code, response_data=res_json)

    try:
        return response.json()
    except Exception as exc:
        raise AIOSKernelError(f"Failed to parse JSON response from AIOS kernel: {exc}") from exc
