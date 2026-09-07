"""
Data Archivist Agent implementation for AIOS Vectros SDK.

Demonstrates:
- Subclassing `BaseAgent`
- Comprehensive Storage API integration (`mount`, `create_dir`, `create_file`, `write_file`, `retrieve_file`, `rollback_file`, `share_file`)
- Data version management and recovery
- Cross-agent file sharing
- Storage event notification via Post API topic channels
"""

from typing import Any, Dict, List, Optional

from vectros_sdk.agent.base import BaseAgent
from vectros_sdk.client.client import AIOSClient
from vectros_sdk.client.config import aios_kernel_url


class DataArchivistAgent(BaseAgent):
    """
    Agent specialized in persistent data storage, directory tree structuring,
    file versioning, corruption rollbacks, and secure cross-agent sharing.
    """

    def __init__(
        self,
        agent_name: str = "data_archivist",
        name: Optional[str] = None,
        client: Optional[AIOSClient] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the Data Archivist Agent.

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
            system_prompt="You are a Data Archivist responsible for file systems, version control, and data integrity.",
            base_url=effective_base_url,
        )
        self.name = self.agent_name
        self.client = client or AIOSClient(base_url=effective_base_url, agent_name=self.agent_name)

    def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute archival tasks based on input payload.

        Args:
            input_data: Task dictionary or directory name to initialize.

        Returns:
            Dict[str, Any]: Results of the storage workflow.
        """
        if isinstance(input_data, dict):
            action = input_data.get("action", "setup_project")
            project_name = input_data.get("project_name", "workspace")
            file_data = input_data.get("file_data", "Initial Project Data")
        else:
            action = "setup_project"
            project_name = str(input_data)
            file_data = "Initial Project Data"

        if action == "setup_project":
            return self.setup_project_storage(project_name=project_name, initial_content=file_data)

        return {"error": f"Unknown action {action}"}

    def setup_project_storage(
        self,
        project_name: str,
        initial_content: str,
        target_collaborator: Optional[str] = None,
        topic: str = "storage_events",
    ) -> Dict[str, Any]:
        """
        Executes a complete storage lifecycle demonstration:
        1. Mounts storage root partition (`client.storage.mount`).
        2. Creates directory hierarchy (`client.storage.create_dir`).
        3. Initializes project files (`client.storage.create_file`).
        4. Writes updated version (`client.storage.write_file`).
        5. Retrieves content with keyword filtering (`client.storage.retrieve_file`).
        6. Rolls back to version 1 on demand (`client.storage.rollback_file`).
        7. Shares the document with a collaborator (`client.storage.share_file`).
        8. Emits a notification to the storage pub/sub topic (`client.post.publish`).

        Args:
            project_name: Name of the project directory.
            initial_content: Initial content for main report file.
            target_collaborator: Optional agent name to share files with.
            topic: Pub/Sub topic channel to publish storage events to.

        Returns:
            Dict[str, Any]: Detailed ledger of all storage operations and statuses.
        """
        log: Dict[str, Any] = {"agent": self.agent_name, "project": project_name, "operations": {}}

        # 1. Mount storage partition
        mount_resp = self.client.storage.mount(root_dir=f"/mnt/{project_name}")
        log["operations"]["mount"] = mount_resp.get("response_message", "Mounted")

        # 2. Create directory
        dir_path = f"/mnt/{project_name}/reports"
        dir_resp = self.client.storage.create_dir(dir_path=dir_path)
        log["operations"]["create_dir"] = dir_resp.get("response_message", f"Created {dir_path}")

        # 3. Create file
        file_path = f"{dir_path}/summary.txt"
        create_resp = self.client.storage.create_file(file_path=file_path)
        log["operations"]["create_file"] = create_resp.get("response_message", f"Created {file_path}")

        # 4. Write new version (v2)
        v2_content = f"{initial_content}\n[Update]: Added supplemental analysis section."
        write_resp = self.client.storage.write_file(
            file_path=file_path,
            content=v2_content,
        )
        log["operations"]["write_file_v2"] = write_resp.get("response_message", "Updated to v2")

        # 5. Retrieve content
        retrieve_resp = self.client.storage.retrieve_file(
            query_text=file_path,
            n=1,
            keywords=["Update", "analysis"],
        )
        log["operations"]["retrieve_file"] = retrieve_resp.get("response_message") or retrieve_resp.get("content")

        # 6. Rollback 1 version
        rollback_resp = self.client.storage.rollback_file(
            file_path=file_path,
            n=1,
        )
        log["operations"]["rollback_v1"] = rollback_resp.get("response_message", "Rolled back to v1")

        # 7. Share file
        share_resp = self.client.storage.share_file(file_path=file_path)
        log["operations"]["share_file"] = share_resp.get("response_message", f"Shared {file_path}")

        # 8. Publish notification to pub/sub topic
        pub_resp = self.client.post.publish(
            topic=topic,
            message={
                "event": "storage_ready",
                "file_path": file_path,
                "project": project_name,
                "collaborator": target_collaborator,
            },
        )
        log["operations"]["topic_publish"] = pub_resp.get("response_message", f"Published to {topic}")

        return log
