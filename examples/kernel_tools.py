# Mix local tools with tools the admin registered in the kernel (incl. MCP).
from vectros import Agent

agent = Agent("researcher", tools=["search_web"], session="research")
for event in agent.stream("Find the latest Linux kernel release."):
    print(event.kind, event.data)
