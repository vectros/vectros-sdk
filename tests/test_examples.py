"""
Test Suite for Example Agents, Custom Tools, and Multi-Agent Demonstration.
Tests execute REAL HTTP network calls against a live local AIOS Kernel server.
"""

import json
from typing import Any, Dict
import pytest

from vectros_sdk.agent.registry import (
    AGENT_REGISTRY,
    clear_agent_registry,
    get_agent,
    list_registered_agents,
    register_agent,
)
from vectros_sdk.client.client import AIOSClient
from vectros_sdk.tool.core.registry import (
    TOOL_REGISTRY,
    clear_registry,
    get_tool,
    list_registered_tools,
    register_tool,
)

from vectros_sdk.examples.tools.custom_tools import (
    DataFormatterTool,
    MathEvaluatorTool,
    SentimentAnalyzerTool,
    register_custom_tools,
)
from vectros_sdk.examples.agents.research_agent import ResearchAnalystAgent
from vectros_sdk.examples.agents.archivist_agent import DataArchivistAgent
from vectros_sdk.examples.agents.coordinator_agent import TaskCoordinatorAgent
from vectros_sdk.examples.server.mock_kernel_server import LiveAIOSKernelServer, start_kernel_server
from vectros_sdk.examples.demo_multi_agent import run_demonstration


class TestCustomTools:
    """Test custom tool implementations."""

    def test_math_evaluator_tool(self) -> None:
        tool = MathEvaluatorTool()
        assert tool.name == "math_evaluator"
        schema = tool.get_schema()
        assert schema["type"] == "function"
        assert "operation" in schema["function"]["parameters"]["properties"]

        # Eval
        res_eval = tool.execute(operation="eval", expression="sqrt(100) + 5 * 2")
        assert res_eval["success"] is True
        assert res_eval["result"] == 20.0

        # Stats
        numbers = [10.0, 20.0, 30.0, 40.0]
        assert tool.execute(operation="sum", numbers=numbers)["result"] == 100.0
        assert tool.execute(operation="average", numbers=numbers)["result"] == 25.0
        assert tool.execute(operation="min", numbers=numbers)["result"] == 10.0
        assert tool.execute(operation="max", numbers=numbers)["result"] == 40.0
        assert tool.execute(operation="variance", numbers=numbers)["success"] is True

        # Error handling
        assert tool.execute(operation="eval", expression="")["success"] is False
        assert tool.execute(operation="sum", numbers=[])["success"] is False
        assert tool.execute(operation="invalid_op")["success"] is False

    def test_data_formatter_tool(self) -> None:
        tool = DataFormatterTool()
        assert tool.name == "data_formatter"
        data = [
            {"Name": "Alice", "Role": "Engineer"},
            {"Name": "Bob", "Role": "Designer"},
        ]

        # Markdown table
        md_res = tool.execute(format_type="markdown_table", data=data)
        assert md_res["success"] is True
        assert "| Name | Role |" in md_res["formatted"]
        assert "| Alice | Engineer |" in md_res["formatted"]

        # CSV
        csv_res = tool.execute(format_type="csv", data=data)
        assert csv_res["success"] is True
        assert "Name,Role" in csv_res["formatted"]
        assert "Alice,Engineer" in csv_res["formatted"]

        # JSON
        json_res = tool.execute(format_type="pretty_json", data=data)
        assert json_res["success"] is True
        parsed = json.loads(json_res["formatted"])
        assert len(parsed) == 2

        # Error cases
        assert tool.execute(format_type="markdown_table", data=[])["success"] is False
        assert tool.execute(format_type="unknown_fmt", data=data)["success"] is False

    def test_sentiment_analyzer_tool(self) -> None:
        tool = SentimentAnalyzerTool()
        assert tool.name == "sentiment_analyzer"

        pos_res = tool.execute(text="This system has great performance, excellent throughput and passed all tests.")
        assert pos_res["success"] is True
        assert pos_res["sentiment"] == "positive"
        assert pos_res["score"] > 0

        neg_res = tool.execute(text="The database encountered a critical error, crash, and corrupted data.")
        assert neg_res["success"] is True
        assert neg_res["sentiment"] == "negative"
        assert neg_res["score"] < 0

        neu_res = tool.execute(text="The system executed the command.")
        assert neu_res["success"] is True
        assert neu_res["sentiment"] == "neutral"

        assert tool.execute(text="")["success"] is False

    def test_register_custom_tools_helper(self) -> None:
        custom_dict: Dict[str, Any] = {}
        register_custom_tools(custom_dict)
        assert "math_evaluator" in custom_dict
        assert "data_formatter" in custom_dict
        assert "sentiment_analyzer" in custom_dict

        # Register to global
        register_custom_tools()
        assert get_tool("math_evaluator") is not None
        assert get_tool("data_formatter") is not None
        assert get_tool("sentiment_analyzer") is not None


class TestExampleAgents:
    """Test concrete example agent implementations using REAL HTTP requests."""

    @pytest.fixture
    def live_server_url(self) -> str:
        srv, url = start_kernel_server(host="127.0.0.1", port=0)
        yield url
        srv.stop()

    @pytest.fixture
    def live_client(self, live_server_url: str) -> AIOSClient:
        return AIOSClient(base_url=live_server_url)

    def test_research_analyst_agent(self, live_client: AIOSClient) -> None:
        agent = ResearchAnalystAgent(name="test_analyst", client=live_client)
        assert agent.name == "test_analyst"

        # Test conduct_research over real HTTP
        res = agent.conduct_research(
            topic="Distributed Caching Performance",
            numbers=[100, 200, 300],
            coordinator_name="test_coordinator",
        )
        assert res["agent"] == "test_analyst"
        assert res["topic"] == "Distributed Caching Performance"
        assert "statistics" in res
        assert "synthesis" in res
        assert res["saved_memory_id"] is not None

        # Test agent.run() interface
        run_res = agent.run({"topic": "AIOS Kernel", "numbers": [10, 20]})
        assert run_res["agent"] == "test_analyst"

    def test_data_archivist_agent(self, live_client: AIOSClient) -> None:
        agent = DataArchivistAgent(name="test_archivist", client=live_client)
        assert agent.name == "test_archivist"

        log = agent.setup_project_storage(
            project_name="project_alpha",
            initial_content="Initial log content.",
            target_collaborator="test_analyst",
            topic="storage_feed",
        )
        assert log["agent"] == "test_archivist"
        assert log["project"] == "project_alpha"
        assert "mount" in log["operations"]
        assert "create_dir" in log["operations"]
        assert "create_file" in log["operations"]
        assert "write_file_v2" in log["operations"]
        assert "retrieve_file" in log["operations"]
        assert "rollback_v1" in log["operations"]
        assert "share_file" in log["operations"]
        assert "topic_publish" in log["operations"]

        # Test agent.run() interface
        run_res = agent.run({"action": "setup_project", "project_name": "workspace_test"})
        assert run_res["project"] == "workspace_test"

    def test_task_coordinator_agent(self, live_client: AIOSClient) -> None:
        agent = TaskCoordinatorAgent(name="test_coordinator", client=live_client)
        assert agent.name == "test_coordinator"

        pipeline_log = agent.orchestrate_pipeline(
            mission="Deploy AIOS Cluster v2",
            worker_names=["worker_1", "worker_2"],
            topic_channel="cluster_events",
        )
        assert pipeline_log["coordinator"] == "test_coordinator"
        assert pipeline_log["mission"] == "Deploy AIOS Cluster v2"
        assert len(pipeline_log["steps"]) == 6

        # Test agent.run() interface
        run_res = agent.run("New Goal")
        assert run_res["mission"] == "New Goal"


class TestFullDemonstrationRunner:
    """Test running the full multi-agent showcase script with real HTTP calls."""

    def test_run_demonstration(self, capsys: pytest.CaptureFixture) -> None:
        run_demonstration(auto_start_server=True)
        captured = capsys.readouterr()
        assert "STARTING REAL AIOS KERNEL HTTP SERVER ON LOCALHOST" in captured.out
        assert "INITIALIZING AIOS CLIENT & REGISTERING CUSTOM TOOLS" in captured.out
        assert "PHASE 1: RESEARCH ANALYST AGENT" in captured.out
        assert "PHASE 2: DATA ARCHIVIST AGENT" in captured.out
        assert "PHASE 3: TASK COORDINATOR AGENT" in captured.out
        assert "PHASE 4: UNIFIED AIOSCLIENT HIGH-LEVEL HELPERS" in captured.out
        assert "VERIFICATION OF REAL HTTP NETWORK TRANSACTIONS" in captured.out
        assert "ALL SDK FEATURES & MULTI-AGENT WORKFLOWS SUCCESSFULLY DEMONSTRATED WITH REAL HTTP REQUESTS!" in captured.out
