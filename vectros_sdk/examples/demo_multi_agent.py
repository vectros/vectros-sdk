"""
AIOS Vectros SDK - Complete Feature & Multi-Agent Demonstration.

This script demonstrates the end-to-end capabilities of the AIOS Vectros SDK:
1. LLM Core API (chat, structured JSON, tool calls)
2. Memory API (CRUD, semantic vector search, agentic memory)
3. Storage API (mount, create_dir, create_file, write_file, retrieve_file, rollback_file, share_file)
4. Tool API & Extensibility (BaseTool subclasses, TOOL_REGISTRY, dynamic execution)
5. Post API (direct messaging, inbox retrieval, system broadcasts, topic pub/sub)
6. Agent API (BaseAgent subclasses, AGENT_REGISTRY)
7. Unified AIOSClient Orchestrator (scoped sub-clients & high-level convenience methods)
"""

import json
import sys
from typing import Any, Dict

# Import Vectros SDK components
from vectros_sdk.client.client import AIOSClient, CerebrumClient
import vectros_sdk.client.send_request
import vectros_sdk.llm.api
import vectros_sdk.memory.api
import vectros_sdk.storage.api
import vectros_sdk.tool.api
import vectros_sdk.post.api

from vectros_sdk.agent.registry import (
    AGENT_REGISTRY,
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

# Import custom example agents and tools
from vectros_sdk.examples.tools.custom_tools import (
    DataFormatterTool,
    MathEvaluatorTool,
    SentimentAnalyzerTool,
    register_custom_tools,
)
from vectros_sdk.examples.agents.research_agent import ResearchAnalystAgent
from vectros_sdk.examples.agents.archivist_agent import DataArchivistAgent
from vectros_sdk.examples.agents.coordinator_agent import TaskCoordinatorAgent


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


def build_mock_client() -> AIOSClient:
    """
    Construct an AIOSClient with simulated kernel responses to allow offline
    demonstration of all SDK APIs without requiring a live running kernel.
    """
    client = AIOSClient(base_url="http://localhost:8000")

    def mock_send_request(query: Any, base_url: str = "", timeout: int = 60) -> Dict[str, Any]:
        q_class = getattr(query, "query_class", query.get("query_class") if isinstance(query, dict) else "unknown")
        action = getattr(query, "action_type", query.get("action_type") if isinstance(query, dict) else "unknown")

        if q_class == "llm":
            return {
                "response": {
                    "response_message": json.dumps({
                        "summary": "AIOS Kernel optimization trends show a 42% throughput gain under concurrent multi-agent loads.",
                        "confidence": 0.98,
                        "key_findings": [
                            "Memory layer caching decreases retrieval latency by 65%",
                            "Storage rollback operations maintain 100% integrity across concurrent revisions",
                            "Pub/sub topic channels scale linearly with active subscriber agents",
                        ],
                        "recommendation": "Adopt AIOSClient as the primary orchestration pattern for production agents.",
                    }),
                    "finished": True,
                    "status_code": 200,
                }
            }
        elif q_class == "memory":
            if action == "search":
                return {
                    "response": {
                        "results": [
                            {"memory_id": "mem_001", "content": "Initial baseline benchmark for AIOS kernel throughput.", "score": 0.91},
                            {"memory_id": "mem_002", "content": "Historical memory index parameters and tuning rules.", "score": 0.87},
                        ],
                        "status_code": 200,
                    }
                }
            elif action == "create_agentic":
                return {
                    "response": {
                        "memory_id": "mem_agentic_987",
                        "response_message": "Agentic memory stored with semantic embeddings and metadata.",
                        "status_code": 200,
                    }
                }
            return {
                "response": {
                    "memory_id": "mem_100",
                    "response_message": f"Memory {action} operation successful.",
                    "status_code": 200,
                }
            }
        elif q_class == "storage":
            return {
                "response": {
                    "response_message": f"Storage operation '{action}' succeeded on target path.",
                    "file_path": getattr(query, "file_path", None),
                    "version": getattr(query, "version", 1),
                    "content": "Simulated retrieved content for AIOS storage demonstration.",
                    "status_code": 200,
                }
            }
        elif q_class == "tool":
            tool_calls = getattr(query, "tool_calls", query.get("tool_calls", []) if isinstance(query, dict) else [])
            results = []
            for tc in tool_calls:
                t_name = tc.get("name")
                params = tc.get("parameters", {})
                if t_name in TOOL_REGISTRY:
                    t_cls = TOOL_REGISTRY[t_name]
                    t_inst = t_cls() if callable(t_cls) else t_cls
                    res = t_inst.execute(**params)
                    results.append(res)
                else:
                    results.append({"status": "success", "tool": t_name})
            msg = json.dumps(results[0]) if len(results) == 1 else json.dumps(results)
            return {
                "response": {
                    "response_message": msg,
                    "result": results[0] if len(results) == 1 else results,
                    "status_code": 200,
                }
            }
        elif q_class == "post":
            if action == "receive":
                return {
                    "response": {
                        "messages": [
                            {"message_id": "msg_001", "sender": "research_analyst", "content": "Research completed with confidence 0.98"},
                            {"message_id": "msg_002", "sender": "data_archivist", "content": "Storage archives mounted and verified"},
                        ],
                        "status_code": 200,
                    }
                }
            return {
                "response": {
                    "message_id": f"msg_post_{action}_55",
                    "response_message": f"Post '{action}' operation acknowledged.",
                    "status_code": 200,
                }
            }
        return {"response": {"response_message": "Default OK", "status_code": 200}}

    # Patch dispatchers across all API namespaces
    vectros_sdk.client.send_request.send_request = mock_send_request
    vectros_sdk.llm.api.send_request = mock_send_request
    vectros_sdk.memory.api.send_request = mock_send_request
    vectros_sdk.storage.api.send_request = mock_send_request
    vectros_sdk.tool.api.send_request = mock_send_request
    vectros_sdk.post.api.send_request = mock_send_request

    client.send_request = mock_send_request
    client.llm.send_request = mock_send_request
    client.memory.send_request = mock_send_request
    client.storage.send_request = mock_send_request
    client.tool.send_request = mock_send_request
    client.post.send_request = mock_send_request
    return client


def run_demonstration(use_mock: bool = True) -> None:
    """
    Run complete end-to-end multi-agent demonstration.

    Args:
        use_mock: If True, uses simulated kernel responses for reliable local execution.
    """
    print_banner("1. Initializing AIOS Client & Registering Custom Tools")
    client = build_mock_client() if use_mock else AIOSClient()
    print(f"-> Created AIOSClient connected to: {client.base_url}")
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

    # =========================================================================
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

    # =========================================================================
    print_banner("3. Phase 1: Research Analyst Agent (LLM, Memory & Tool Execution)")
    benchmark_metrics = [120.5, 135.2, 142.8, 118.9, 155.0, 160.4]
    research_report = researcher.conduct_research(
        topic="Kernel Latency & Throughput Optimization",
        numbers=benchmark_metrics,
        coordinator_name="task_coordinator",
    )
    print_json("Research Analyst Execution Report", research_report)

    # =========================================================================
    print_banner("4. Phase 2: Data Archivist Agent (Storage API 7 Operations & Versioning)")
    storage_log = archivist.setup_project_storage(
        project_name="kernel_q3_benchmarks",
        initial_content="AIOS Kernel v0.5.0 Benchmark Performance Analysis Baseline.",
        target_collaborator="research_analyst",
        topic="storage_events",
    )
    print_json("Data Archivist Storage Operations Log", storage_log)

    # =========================================================================
    print_banner("5. Phase 3: Task Coordinator Agent (Post API Pub/Sub & Inbox Messaging)")
    coordination_log = coordinator.orchestrate_pipeline(
        mission="AIOS Kernel Q3 Performance & Storage Milestone",
        worker_names=["research_analyst", "data_archivist"],
        topic_channel="project_milestones",
    )
    print_json("Task Coordinator Pipeline Log", coordination_log)

    # =========================================================================
    print_banner("6. Phase 4: Unified AIOSClient High-Level Helpers")

    # 6.1 Top-level remember and recall
    print("-> 6.1 Calling client.remember() & client.recall():")
    rem_resp = client.remember(
        content="AIOS provides unified multi-agent operating system primitives.",
        metadata={"category": "architecture", "version": "1.0"},
    )
    print(f"   * Remembered memory ID: {rem_resp.get('memory_id')}")

    rec_resp = client.recall(query="operating system primitives", k=2)
    print(f"   * Recalled results count: {len(rec_resp.get('results', []))}")

    # 6.2 Top-level direct messaging
    print("\n-> 6.2 Calling client.send_message():")
    post_resp = client.send_message(
        recipient="research_analyst",
        message="Please compile final executive slides.",
        metadata={"priority": "high"},
    )
    print(f"   * Message dispatched with ID: {post_resp.get('message_id')}")

    # 6.3 Top-level chat
    print("\n-> 6.3 Calling client.chat():")
    chat_resp = client.chat(
        prompt="Explain why AIOS kernel isolation improves multi-agent stability.",
        system_prompt="You are an AI Systems Architect.",
    )
    print(f"   * Chat response excerpt: {str(chat_resp.response_message)[:120]}...")

    print_banner("All SDK Features & Multi-Agent Workflows Successfully Demonstrated!")


if __name__ == "__main__":
    is_live = "--live" in sys.argv
    run_demonstration(use_mock=not is_live)
