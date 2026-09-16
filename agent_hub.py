import asyncio
import json
import logging
from aiohttp import web

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentHub")

class AgentHub:
    def __init__(self):
        # Registry mapping Agent ID to their WebSocket connections (AUM/ADM <-> AHM)
        self.agents = {}
        # Mapping for Agent Runtime Machines (ARM) which have the AIOS Kernel
        self.runtime_machines = {}

    async def handle_ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        agent_id = request.query.get('agent_id', 'unknown')
        machine_type = request.query.get('type', 'AUM') # AUM, ARM, or ADM
        
        logger.info(f"New connection from {agent_id} (Type: {machine_type})")
        
        if machine_type == 'ARM':
            self.runtime_machines[agent_id] = ws
        else:
            self.agents[agent_id] = ws

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    target = data.get("target")
                    if target in self.agents:
                        logger.info(f"Routing message from {agent_id} to {target}")
                        await self.agents[target].send_str(msg.data)
                    elif target in self.runtime_machines:
                        logger.info(f"Delegating workload to ARM {target}")
                        await self.runtime_machines[target].send_str(msg.data)
                    else:
                        await ws.send_str(json.dumps({"error": f"Target {target} not found in Agent Hub"}))
        finally:
            if machine_type == 'ARM':
                self.runtime_machines.pop(agent_id, None)
            else:
                self.agents.pop(agent_id, None)
            logger.info(f"Agent {agent_id} disconnected.")
            
        return ws

async def init_app():
    app = web.Application()
    hub = AgentHub()
    app.router.add_get('/ws', hub.handle_ws)
    return app

if __name__ == '__main__':
    web.run_app(init_app(), port=8080)
