"""A DeepAgent that plans, keeps notes and delegates research."""

from vectros import DeepAgent, MCPServer, SubAgent, ask_terminal

files = MCPServer("files", "npx -y @modelcontextprotocol/server-filesystem .")
researcher = SubAgent("researcher", "Reads local files and reports facts.",
                      system="Report only facts you read.", tools=[files])
agent = DeepAgent("analyst", subagents=[researcher], approve=ask_terminal)

if __name__ == "__main__":
    print(agent.run("Summarize the README in this directory in five bullet points."))
    print(agent.todos)
