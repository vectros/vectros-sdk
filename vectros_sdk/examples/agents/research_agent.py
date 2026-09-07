"""
Research Analyst Agent implementation for AIOS Vectros SDK.

Demonstrates:
- Subclassing `BaseAgent`
- Using `AIOSClient` with scoped sub-clients (`.memory`, `.tool`, `.llm`, `.post`)
- Semantic memory retrieval and episodic memory creation
- Custom tool execution
- Structured JSON output generation
"""

from typing import Any, Dict, List, Optional

from vectros_sdk.agent.base import BaseAgent
from vectros_sdk.client.client import AIOSClient
from vectros_sdk.client.config import aios_kernel_url


class ResearchAnalystAgent(BaseAgent):
    """
    Agent specialized in researching topics, recalling domain memories,
    executing quantitative tools, synthesizing insights with LLMs, and saving
    structured findings.
    """

    def __init__(
        self,
        agent_name: str = "research_analyst",
        name: Optional[str] = None,
        client: Optional[AIOSClient] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the Research Analyst Agent.

        Args:
            agent_name: Primary agent namespace identifier.
            name: Optional alias for agent_name.
            client: Optional pre-configured AIOSClient instance.
            base_url: Optional kernel API URL override.
        """
        chosen_name = name or agent_name
        effective_base_url = base_url or (client.base_url if client else aios_kernel_url)
        super().__init__(
            agent_name=chosen_name,
            system_prompt="You are a Senior Quantitative Research Analyst specializing in systems optimization.",
            base_url=effective_base_url,
        )
        self.name = self.agent_name
        self.client = client or AIOSClient(base_url=effective_base_url, agent_name=self.agent_name)

    def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute research workflow on the provided topic or query dictionary.

        Args:
            input_data: Topic string or dictionary containing query details.

        Returns:
            Dict[str, Any]: Structured research findings and memory IDs.
        """
        if isinstance(input_data, str):
            topic = input_data
            numbers = []
        elif isinstance(input_data, dict):
            topic = input_data.get("topic", "General Analysis")
            numbers = input_data.get("numbers", [])
        else:
            topic = str(input_data)
            numbers = []

        return self.conduct_research(topic=topic, numbers=numbers)

    def conduct_research(
        self,
        topic: str,
        numbers: Optional[List[float]] = None,
        coordinator_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full research lifecycle:
        1. Search previous memories for related context (`client.recall`).
        2. Perform calculations with math tool if numerical data is provided.
        3. Synthesize conclusions using structured LLM (`client.llm.chat_json`).
        4. Store structured insights in agentic memory (`client.memory.create_agentic`).
        5. Optionally notify coordinator via Post API.

        Args:
            topic: Research subject.
            numbers: Optional list of numeric values to analyze statistically.
            coordinator_name: Optional coordinator agent to report results to.

        Returns:
            Dict[str, Any]: Research report with synthesis, metrics, and memory ID.
        """
        # 1. Memory recall
        recalled_memories = self.client.recall(query=topic, k=3)
        prior_context = [
            m.get("content", "") for m in recalled_memories.get("results", []) if isinstance(m, dict)
        ]

        # 2. Tool calculation (if numbers provided)
        stats: Dict[str, Any] = {}
        if numbers:
            math_resp = self.client.tool.call(
                tool_calls=[{
                    "name": "math_evaluator",
                    "parameters": {"operation": "average", "numbers": numbers},
                }]
            )
            stats["average"] = math_resp.get("response_message") or math_resp.get("result")

            var_resp = self.client.tool.call(
                tool_calls=[{
                    "name": "math_evaluator",
                    "parameters": {"operation": "variance", "numbers": numbers},
                }]
            )
            stats["variance"] = var_resp.get("response_message") or var_resp.get("result")

        # 3. LLM synthesis
        prompt = (
            f"Analyze research topic: '{topic}'.\n"
            f"Prior Context: {prior_context}\n"
            f"Statistical Metrics: {stats}\n"
            "Provide key findings, confidence score, and recommendations."
        )
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "confidence": {"type": "number"},
                "key_findings": {"type": "array", "items": {"type": "string"}},
                "recommendation": {"type": "string"},
            },
            "required": ["summary", "key_findings"],
        }
        llm_resp = self.client.llm.chat_json(
            messages=[
                {"role": "system", "content": "You are a Senior Quantitative Research Analyst."},
                {"role": "user", "content": prompt},
            ],
            response_format=schema,
        )
        synthesis = llm_resp.response_message or str(llm_resp.raw_response)

        # 4. Save to Agentic Memory
        mem_resp = self.client.memory.create_agentic(
            content=f"Research on '{topic}': {synthesis}",
            metadata={"tags": ["research", topic.lower().replace(" ", "_")], "stats": stats, "confidence": 0.95},
        )
        memory_id = mem_resp.get("memory_id")

        result = {
            "agent": self.agent_name,
            "topic": topic,
            "prior_memories_found": len(prior_context),
            "statistics": stats,
            "synthesis": synthesis,
            "saved_memory_id": memory_id,
        }

        # 5. Direct Post notification to coordinator if requested
        if coordinator_name:
            self.client.post.send(
                recipient=coordinator_name,
                message={"status": "complete", "topic": topic, "summary": synthesis},
            )

        return result
