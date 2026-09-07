"""
Task Coordinator Agent implementation for AIOS Vectros SDK.

Demonstrates:
- Subclassing `BaseAgent`
- Multi-agent orchestration using Post API:
  - `send_post`: Direct task delegation to worker agents
  - `receive_posts`: Polling inbox for task completion reports
  - `publish_to_topic` / `subscribe_topic`: Topic-based event channels
  - `broadcast_post`: System-wide broadcasts
- Team orchestration lifecycle
"""

from typing import Any, Dict, List, Optional, Type

from vectros_sdk.agent.base import BaseAgent
from vectros_sdk.agent.registry import AGENT_REGISTRY, get_agent, list_registered_agents, register_agent
from vectros_sdk.client.client import AIOSClient
from vectros_sdk.client.config import aios_kernel_url


class TaskCoordinatorAgent(BaseAgent):
    """
    Orchestrator agent that breaks down high-level project goals, dispatches
    work packages to specialized agents via direct Post messaging, monitors pub/sub
    channels, collects inbox reports, and broadcasts project completions.
    """

    def __init__(
        self,
        agent_name: str = "task_coordinator",
        name: Optional[str] = None,
        client: Optional[AIOSClient] = None,
        registry: Optional[Dict[str, Type[Any]]] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the Task Coordinator Agent.

        Args:
            agent_name: Primary agent namespace identifier.
            name: Optional alias for agent_name.
            client: Optional pre-configured AIOSClient instance.
            registry: Optional Agent registry dictionary.
            base_url: Optional kernel API URL override.
        """
        chosen_name = name or agent_name
        effective_base_url = base_url or (client.base_url if client else aios_kernel_url)
        super().__init__(
            agent_name=chosen_name,
            system_prompt="You are a Task Coordinator Agent responsible for orchestrating multi-agent pipelines.",
            base_url=effective_base_url,
        )
        self.name = self.agent_name
        self.description = "Coordinates multi-agent task pipelines, dispatches sub-tasks, collects inbox messages, and broadcasts milestones."
        self.client = client or AIOSClient(base_url=effective_base_url, agent_name=self.agent_name)
        self.registry = registry or AGENT_REGISTRY

    def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute coordination workflow for a given mission.

        Args:
            input_data: Mission string or configuration dictionary.

        Returns:
            Dict[str, Any]: Execution pipeline results.
        """
        if isinstance(input_data, dict):
            mission = input_data.get("mission", "Quarterly AIOS Project Analysis")
            workers = input_data.get("workers", ["research_analyst", "data_archivist"])
        else:
            mission = str(input_data)
            workers = ["research_analyst", "data_archivist"]

        return self.orchestrate_pipeline(mission=mission, worker_names=workers)

    def orchestrate_pipeline(
        self,
        mission: str,
        worker_names: List[str],
        topic_channel: str = "project_milestones",
    ) -> Dict[str, Any]:
        """
        Orchestrate complete multi-agent workflow:
        1. Subscribe to topic channel (`client.post.subscribe`).
        2. Broadcast initial task kickoff notice (`client.post.broadcast`).
        3. Dispatch discrete tasks to worker agents via direct messages (`client.post.send`).
        4. Poll inbox for responses (`client.post.receive`).
        5. Publish milestone accomplishment on topic (`client.post.publish`).
        6. Issue final system-wide completion broadcast (`client.post.broadcast`).

        Args:
            mission: Project mission description.
            worker_names: Names of registered worker agents to dispatch tasks to.
            topic_channel: Pub/Sub topic for milestone announcements.

        Returns:
            Dict[str, Any]: Consolidated multi-agent pipeline execution report.
        """
        execution_log: Dict[str, Any] = {
            "coordinator": self.agent_name,
            "mission": mission,
            "workers": worker_names,
            "steps": [],
        }

        # 1. Subscribe to topic
        sub_resp = self.client.post.subscribe(topic=topic_channel)
        execution_log["steps"].append({
            "action": "subscribe_topic",
            "topic": topic_channel,
            "status": sub_resp.get("response_message", "Subscribed"),
        })

        # 2. Broadcast mission kickoff
        kickoff_msg = f"[KICKOFF] Mission '{mission}' initiated by coordinator {self.agent_name}."
        bcast_resp = self.client.post.broadcast(
            message=kickoff_msg,
            topic=topic_channel,
        )
        execution_log["steps"].append({
            "action": "broadcast_kickoff",
            "message": kickoff_msg,
            "status": bcast_resp.get("response_message", "Broadcasted"),
        })

        # 3. Dispatch specific tasks to each worker via direct Post messages
        dispatched_tasks = []
        for worker in worker_names:
            task_payload = {
                "mission": mission,
                "assigned_worker": worker,
                "instructions": f"Execute required research or storage workflows for {mission}.",
            }
            send_resp = self.client.post.send(
                recipient=worker,
                message=task_payload,
            )
            dispatched_tasks.append({
                "worker": worker,
                "message_id": send_resp.get("message_id"),
                "status": send_resp.get("response_message", "Sent"),
            })

        execution_log["steps"].append({
            "action": "dispatch_tasks",
            "tasks": dispatched_tasks,
        })

        # 4. Check inbox for worker feedback
        inbox_resp = self.client.post.receive(limit=10, mark_as_read=True)
        received_messages = inbox_resp.get("messages", [])
        execution_log["steps"].append({
            "action": "check_inbox",
            "messages_count": len(received_messages),
            "messages": received_messages,
        })

        # 5. Publish milestone to channel
        milestone_payload = {
            "status": "in_progress",
            "mission": mission,
            "dispatched_count": len(worker_names),
        }
        pub_resp = self.client.post.publish(
            topic=topic_channel,
            message=milestone_payload,
        )
        execution_log["steps"].append({
            "action": "publish_milestone",
            "topic": topic_channel,
            "status": pub_resp.get("response_message", "Published"),
        })

        # 6. Final broadcast completion
        final_bcast = self.client.post.broadcast(
            message=f"[COMPLETE] Mission '{mission}' successfully coordinated.",
        )
        execution_log["steps"].append({
            "action": "broadcast_complete",
            "status": final_bcast.get("response_message", "Broadcast complete"),
        })

        return execution_log
