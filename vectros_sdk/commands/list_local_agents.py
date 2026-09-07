import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from vectros_sdk.agent.registry import list_registered_agents


def list_local_agents(agents_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List agents installed locally on the system and registered in the local agent registry.

    Args:
        agents_dir: Optional root directory path to scan for agent config.json files.

    Returns:
        List[Dict[str, Any]]: List of local agent info dictionaries.
    """
    agents = []
    seen_names = set()

    # 1. Registered agents in memory registry
    reg_agents = list_registered_agents()
    for name, agent_cls in reg_agents.items():
        doc = getattr(agent_cls, "__doc__", "") or "Registered Python agent class"
        agents.append({
            "name": name,
            "type": "registered_class",
            "class": getattr(agent_cls, "__name__", str(agent_cls)),
            "description": doc.strip(),
            "source": "AGENT_REGISTRY",
        })
        seen_names.add(name)

    # 2. Filesystem agents scanning
    scan_paths = []
    if agents_dir:
        scan_paths.append(Path(agents_dir))
    else:
        # Default scan in vectros_sdk/agent/core
        core_dir = Path(__file__).parent.parent / "agent" / "core"
        if core_dir.exists():
            scan_paths.append(core_dir)

    for base_dir in scan_paths:
        if not base_dir.exists():
            continue
        for config_path in base_dir.glob("**/config.json"):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                name = cfg.get("name")
                if name and name not in seen_names:
                    meta = cfg.get("meta", {})
                    author = meta.get("author", "local")
                    version = meta.get("version", "0.0.1")
                    desc = cfg.get("description", "")
                    if isinstance(desc, list):
                        desc = " ".join(desc)
                    agents.append({
                        "name": name,
                        "author": author,
                        "version": version,
                        "description": desc,
                        "path": str(config_path.parent),
                        "source": "filesystem",
                    })
                    seen_names.add(name)
            except Exception:
                continue

    return agents


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="List agents installed on local system.")
    parser.add_argument(
        "--agents_dir",
        default=None,
        help="Custom directory path to scan for local agents",
    )
    parsed = parser.parse_args(args)

    agents = list_local_agents(agents_dir=parsed.agents_dir)
    if not agents:
        print("No local agents found.")
        return 0

    print(f"=== Locally Available Agents ({len(agents)}) ===")
    for agent in agents:
        name = agent.get("name", "Unknown")
        author = agent.get("author", "N/A")
        version = agent.get("version", "")
        ver_str = f" (v{version})" if version else ""
        desc = agent.get("description", "")
        source = agent.get("source", "local")
        print(f"- {name}{ver_str} [{source}]: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
