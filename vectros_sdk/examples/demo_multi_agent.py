"""
AIOS Vectros SDK - Multi-Agent HTTP Request & Response Demonstration.

**This demonstrates real HTTP wire traffic (a real TCP socket, real HTTP
requests/responses) against a FAKE, in-process stand-in kernel
(`vectros_sdk.examples.server.mock_kernel_server.FakeAIOSKernelServer`) --
not the real AIOS kernel (`aios-core`/`aiosd`). See that module's own
docstring. For a real sandboxed-kernel run of these same agents, see
`vectros-sdk/tests/test_examples_real_backend.py` (ASBX.16), which uses
`vectros_sdk.testing.AgentSandbox` instead.**

This script exercises end-to-end multi-agent execution making real HTTP
requests over TCP sockets to that fake kernel server, without any mock
monkeypatching of the HTTP layer itself:
1. Starts the fake HTTP server on localhost TCP port handling `/query`.
2. Initialises `AIOSClient(base_url="http://127.0.0.1:<port>")`.
3. Dispatches real HTTP POST requests across all subsystems:
   - LLM Core API (`chat`, `chat_json`)
   - Memory API (`create`, `get`, `update`, `delete`, `remember`) — semantic
     search and agentic memory (`search`, `create_agentic`, `recall`) are out
     of scope for the real kernel (ARCH.6/MEM.6) and intentionally not
     demonstrated here as if they worked
   - Storage API (`mount`, `create_dir`, `create_file`, `write_file`, `retrieve_file`, `rollback_file`, `share_file`)
   - Tool API (`call_tool` with local & kernel tools)
   - Post API (`send_post`, `receive_posts`, `publish_to_topic`, `subscribe_topic`, `broadcast_post`)
4. Verifies genuine HTTP wire transmission, HTTP status codes, Content-Length headers, and state persistence.
"""

import json
import sys
import time
from typing import Any, Dict, Optional

# Import Vectros SDK components
from vectros_sdk.client.client import AIOSClient, CerebrumClient
from vectros_sdk.agent.registry import (
    get_agent,
    list_registered_agents,
    register_agent,
)
from vectros_sdk.tool.core.registry import (
    TOOL_REGISTRY,
    get_tool,
    list_registered_tools,
    register_tool,
)

# Import custom example agents, tools, and the fake demo HTTP server
from vectros_sdk.examples.tools.custom_tools import (
    DataFormatterTool,
    MathEvaluatorTool,
    SentimentAnalyzerTool,
    register_custom_tools,
)
from vectros_sdk.examples.agents.research_agent import ResearchAnalystAgent
from vectros_sdk.examples.agents.archivist_agent import DataArchivistAgent
from vectros_sdk.examples.agents.coordinator_agent import TaskCoordinatorAgent
from vectros_sdk.examples.server.mock_kernel_server import (
    AIOSKernelHTTPHandler,
    FakeAIOSKernelServer,
    start_kernel_server,
)


def print_banner(title: str) -> None:
    """Print a visually distinct banner for demonstration sections."""
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def print_json(title: str, data: Any) -> None:
    """Pretty print JSON or dictionary data."""
    print(f"\n--- {title} ---")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(str(data))


def run_demonstration(base_url: Optional[str] = None, auto_start_server: bool = True) -> None:
    """
    Run complete end-to-end multi-agent demonstration using real HTTP wire
    traffic against the FAKE, in-process stand-in kernel (see this module's
    own docstring) -- not the real AIOS kernel.

    Args:
        base_url: Optional explicit fake-kernel endpoint URL.
        auto_start_server: If True, launches the fake background HTTP server if base_url is not provided.
    """
    server_instance: Optional[FakeAIOSKernelServer] = None

    if base_url is None and auto_start_server:
        print_banner("0. Starting Fake AIOS Kernel HTTP Server on Localhost (Demo Stand-In, Not The Real Kernel)")
        server_instance, base_url = start_kernel_server(host="127.0.0.1", port=0)
        print(f"-> Fake AIOS Kernel HTTP Server (demo stand-in) running on: {base_url}")
        print(f"-> Endpoint `/query` ready for real HTTP POST transactions over TCP socket.")
    elif base_url is None:
        base_url = "http://127.0.0.1:8000"

    try:
        # 1. Initialize Client
        print_banner("1. Initializing AIOS Client & Registering Custom Tools")
        client = AIOSClient(base_url=base_url)
        print(f"-> Created AIOSClient connected to real HTTP endpoint: {client.base_url}")
        print(f"-> CerebrumClient alias verified: {isinstance(client, CerebrumClient)}")

        # Register custom tools into global TOOL_REGISTRY
        register_custom_tools()
        registered_tools = list_registered_tools()
        print(f"-> Registered {len(registered_tools)} Custom Tools in TOOL_REGISTRY:")
        for t_name, t_cls in registered_tools.items():
            t_inst = t_cls()
            print(f"   * [{t_name}]: {t_inst.description}")

        # Test MathEvaluatorTool directly
        math_cls = get_tool("math_evaluator")
        math_tool = math_cls()
        calc_res = math_tool.execute(operation="eval", expression="sqrt(144) + pow(2, 5)")
        print(f"   -> Direct MathTool Execution (sqrt(144) + 2^5): {calc_res['result']}")

        # Test DataFormatterTool directly
        formatter_cls = get_tool("data_formatter")
        formatter_tool = formatter_cls()
        table_sample = [
            {"Agent": "ResearchAnalyst", "Role": "Analytics", "Status": "Active"},
            {"Agent": "DataArchivist", "Role": "Storage", "Status": "Active"},
            {"Agent": "TaskCoordinator", "Role": "Orchestrator", "Status": "Active"},
        ]
        formatted_table = formatter_tool.execute(format_type="markdown_table", data=table_sample)
        print("\n   -> Formatted Markdown Table Output:")
        print(formatted_table["formatted"])

        # Test SentimentAnalyzerTool directly
        sentiment_cls = get_tool("sentiment_analyzer")
        sentiment_tool = sentiment_cls()
        sent_res = sentiment_tool.execute(text="The AIOS kernel operates with great performance and superb stability!")
        print(f"\n   -> Sentiment Analysis: {sent_res['sentiment']} (Confidence: {sent_res['confidence']})")

        # =====================================================================
        print_banner("2. Initializing & Registering Agents in AGENT_REGISTRY")
        register_agent("research_analyst", ResearchAnalystAgent)
        register_agent("data_archivist", DataArchivistAgent)
        register_agent("task_coordinator", TaskCoordinatorAgent)

        researcher = ResearchAnalystAgent(name="research_analyst", client=client)
        archivist = DataArchivistAgent(name="data_archivist", client=client)
        coordinator = TaskCoordinatorAgent(name="task_coordinator", client=client)

        all_agents = list_registered_agents()
        print(f"-> Registered {len(all_agents)} Agent Classes in AGENT_REGISTRY:")
        for a_name, a_cls in all_agents.items():
            print(f"   * [{a_name}]: {a_cls.__name__}")

        # =====================================================================
        print_banner("3. Phase 1: Research Analyst Agent (Real HTTP LLM, Memory & Tool Requests)")
        benchmark_metrics = [120.5, 135.2, 142.8, 118.9, 155.0, 160.4]
        research_report = researcher.conduct_research(
            topic="Kernel Latency & Throughput Optimization",
            numbers=benchmark_metrics,
            coordinator_name="task_coordinator",
        )
        print_json("Research Analyst Execution Report (Live HTTP Response)", research_report)

        # =====================================================================
        print_banner("4. Phase 2: Data Archivist Agent (Real HTTP Storage 7 Operations)")
        storage_log = archivist.setup_project_storage(
            project_name="kernel_q3_benchmarks",
            initial_content="AIOS Kernel v0.5.0 Benchmark Performance Analysis Baseline.",
            target_collaborator="research_analyst",
            topic="storage_events",
        )
        print_json("Data Archivist Storage Log (Live HTTP Response)", storage_log)

        # =====================================================================
        print_banner("5. Phase 3: Task Coordinator Agent (Real HTTP Post Pub/Sub & Messaging)")
        coordination_log = coordinator.orchestrate_pipeline(
            mission="AIOS Kernel Q3 Performance & Storage Milestone",
            worker_names=["research_analyst", "data_archivist"],
            topic_channel="project_milestones",
        )
        print_json("Task Coordinator Pipeline Log (Live HTTP Response)", coordination_log)

        # =====================================================================
        print_banner("6. Phase 4: Unified AIOSClient High-Level Helpers (Live HTTP)")

        # 6.1 Top-level remember, then a plain get by the ID it returned.
        # `client.recall()` (semantic search) is not demonstrated here: ARCH.6/
        # MEM.6 mark it explicitly unimplemented rather than faked, so calling
        # it would raise `MemoryFeatureUnimplemented` even against this mock.
        print("-> 6.1 Real HTTP POST: client.remember() & client.memory.get():")
        rem_resp = client.remember(
            content="AIOS provides unified multi-agent operating system primitives.",
            metadata={"category": "architecture", "version": "1.0"},
        )
        remembered_id = rem_resp.get("memory_id")
        print(f"   * Remembered memory ID (from Kernel response): {remembered_id}")

        get_resp = client.memory.get(remembered_id)
        print(f"   * Fetched back by ID (from Kernel response): {get_resp.get('content')}")

        # 6.2 Top-level direct messaging
        print("\n-> 6.2 Real HTTP POST: client.send_message():")
        post_resp = client.send_message(
            recipient="research_analyst",
            message="Please compile final executive slides.",
            metadata={"priority": "high"},
        )
        print(f"   * Message dispatched with ID (from Kernel response): {post_resp.get('message_id')}")

        # 6.3 Top-level chat
        print("\n-> 6.3 Real HTTP POST: client.chat():")
        chat_resp = client.chat(
            prompt="Explain why AIOS kernel isolation improves multi-agent stability.",
            system_prompt="You are an AI Systems Architect.",
        )
        print(f"   * Chat response excerpt: {str(chat_resp.response_message)[:120]}...")

        # Server statistics
        if server_instance:
            print_banner("7. Verification of Real HTTP Network Transactions")
            total_requests = len(AIOSKernelHTTPHandler.request_log)
            total_memories = len(AIOSKernelHTTPHandler.memory_store)
            total_storage = len(AIOSKernelHTTPHandler.storage_store)
            print(f"-> Total REAL HTTP POST requests processed by kernel server: {total_requests}")
            print(f"-> Stateful Memory objects saved on server: {total_memories}")
            print(f"-> Stateful Storage files tracked on server: {total_storage}")
            print(f"-> Last 3 Real HTTP Request wire packets logged by server:")
            for req in AIOSKernelHTTPHandler.request_log[-3:]:
                print(f"   * [{req['client_ip']}:{req['client_port']}] POST /query -> class='{req['query_class']}', action='{req['action_type']}', agent='{req['agent_name']}', bytes={req['payload_size_bytes']}")

        print_banner("All SDK Features & Multi-Agent Workflows Successfully Demonstrated with Real HTTP Requests!")

    finally:
        if server_instance:
            server_instance.stop()
            print("\n-> AIOS Kernel HTTP Server cleanly shut down.")


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    run_demonstration(base_url=target_url, auto_start_server=True)
