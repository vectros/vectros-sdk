"""
Configuration module for AIOS Kernel connection settings.

Manages default kernel endpoint URLs and runtime/environment overrides.
"""

import os

DEFAULT_AIOS_KERNEL_URL: str = "http://localhost:8000"
"""str: Default HTTP endpoint for local AIOS kernel instance."""

aios_kernel_url: str = os.getenv("AIOS_KERNEL_URL", DEFAULT_AIOS_KERNEL_URL)
"""str: Active AIOS kernel URL configured for the current runtime session."""


def get_kernel_url() -> str:
    """
    Retrieve the current AIOS kernel endpoint URL.

    Checks the `AIOS_KERNEL_URL` environment variable first, falling back
    to the active module-level `aios_kernel_url` string.

    Returns:
        str: The currently active AIOS kernel URL endpoint.

    Example:
        >>> from vectros_sdk.client.config import get_kernel_url
        >>> url = get_kernel_url()
        >>> print(url)
        'http://localhost:8000'
    """
    return os.getenv("AIOS_KERNEL_URL", aios_kernel_url)


def set_kernel_url(url: str) -> None:
    """
    Set the default AIOS kernel URL globally for the current runtime session.

    Args:
        url (str): The new base endpoint URL for the AIOS kernel (e.g. 'http://remote:8000').

    Example:
        >>> from vectros_sdk.client.config import set_kernel_url
        >>> set_kernel_url("http://192.168.1.100:8000")
    """
    global aios_kernel_url
    aios_kernel_url = url
