import os

DEFAULT_AIOS_KERNEL_URL: str = "http://localhost:8000"

aios_kernel_url: str = os.getenv("AIOS_KERNEL_URL", DEFAULT_AIOS_KERNEL_URL)


def get_kernel_url() -> str:
    """Get the current AIOS kernel URL."""
    return os.getenv("AIOS_KERNEL_URL", aios_kernel_url)


def set_kernel_url(url: str) -> None:
    """Set the default AIOS kernel URL globally in the runtime."""
    global aios_kernel_url
    aios_kernel_url = url
