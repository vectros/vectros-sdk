"""
Framework adapter example demonstrating the operation lifecycle
using the unified KernelClient.
"""

from vectros_sdk.transport.kernel_client import KernelClient
import uuid
import time
from concurrent.futures import ThreadPoolExecutor

def run_operation(client: KernelClient, agent_id: str, instance_id: str, run_id: str, op_id: str):
    # OPERATION_KIND_INFER = 1
    kind_int = 1
    
    print(f"[{op_id}] Submitting operation...")
    client.submit(
        agent_id=agent_id,
        instance_id=instance_id,
        run_id=run_id,
        operation_id=op_id,
        kind_int=kind_int,
        payload='{"prompt": "Hello world"}'
    )
    
    print(f"[{op_id}] Watching events...")
    # Polling watch
    events = client.watch(agent_id, run_id)
    for event in events:
        print(f"[{op_id}] Event: {event['event_type']} at {event.get('timestamp')}")
        if event["event_type"] in ["OPERATION_EVENT_SUCCEEDED", "OPERATION_EVENT_FAILED", "OPERATION_EVENT_CANCELLED"]:
            break
            
    print(f"[{op_id}] Done.")

def main():
    client = KernelClient()
    print(f"Client mode: {client.mode}")
    
    agent_id = "agent:framework-demo"
    instance_id = "inst-" + str(uuid.uuid4())[:8]
    run_id = "run-" + str(uuid.uuid4())[:8]
    
    op1 = "op-" + str(uuid.uuid4())[:8]
    op2 = "op-" + str(uuid.uuid4())[:8]
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(run_operation, client, agent_id, instance_id, run_id, op1)
        executor.submit(run_operation, client, agent_id, instance_id, run_id, op2)

if __name__ == '__main__':
    main()
