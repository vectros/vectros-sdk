"""
Command-line tool and API for listing agents on the AIOS Agent Hub.

Provides `list_agenthub_agents` programmatic function and `main` CLI entry point.
"""

import argparse
import sys
from typing import Any, Dict, List, Optional
import requests

DEFAULT_AGENTHUB_URL = "https://app.aios.foundation"
"""str: Default Agent Hub endpoint URL."""


def list_agenthub_agents(
    agenthub_url: str = DEFAULT_AGENTHUB_URL,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Fetch and list all agents available on the AIOS Agent Hub.

    Args:
        agenthub_url (str, optional): URL endpoint of the Agent Hub. Defaults to DEFAULT_AGENTHUB_URL.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.

    Returns:
        List[Dict[str, Any]]: List of agent metadata dictionaries containing name, author, version, and description.

    Example:
        >>> from vectros_sdk.commands.list_agenthub import list_agenthub_agents
        >>> agents = list_agenthub_agents()
        >>> for a in agents:
        ...     print(a["name"])
    """
    url = agenthub_url.rstrip("/")
    endpoint = f"{url}/agents" if not url.endswith("/agents") else url
    try:
        response = requests.get(endpoint, headers={"Accept": "application/json"}, timeout=timeout)
        if response.ok:
            data = response.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "agents" in data and isinstance(data["agents"], list):
                return data["agents"]
            return [data] if isinstance(data, dict) else []
        print(f"Failed to fetch agenthub agents: HTTP {response.status_code} - {response.text}", file=sys.stderr)
        return []
    except Exception as exc:
        print(f"Error connecting to Agent Hub at {endpoint}: {exc}", file=sys.stderr)
        return []


def main(args: Optional[List[str]] = None) -> int:
    """
    CLI entry point for `list-agenthub-agents`.

    Args:
        args (Optional[List[str]], optional): Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        int: Exit status code (0 for success).
    """
    parser = argparse.ArgumentParser(description="List agents available on AIOS Agent Hub.")
    parser.add_argument(
        "--agenthub_url",
        default=DEFAULT_AGENTHUB_URL,
        help=f"Agent Hub URL (default: {DEFAULT_AGENTHUB_URL})",
    )
    parsed = parser.parse_args(args)

    agents = list_agenthub_agents(agenthub_url=parsed.agenthub_url)
    if not agents:
        print("No agents found on Agent Hub or failed to retrieve.")
        return 0

    print(f"=== Available Agents on Agent Hub ({len(agents)}) ===")
    for agent in agents:
        name = agent.get("name", "Unknown")
        meta = agent.get("meta", {})
        author = meta.get("author", agent.get("author", "Unknown"))
        version = meta.get("version", agent.get("version", "N/A"))
        desc = agent.get("description", "No description")
        if isinstance(desc, list):
            desc = " ".join(desc)
        print(f"- {name} (v{version}) by {author}: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
