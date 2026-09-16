import sys
import os
import json
import uuid
sys.path.insert(0, os.path.dirname(__file__))

from vectros_sdk.transport.grpc_transport import GrpcTransport

def main():
    print("Connecting to [::1]:50051...")
    transport = GrpcTransport(address="[::1]:50051", timeout=300.0)
    
    agent_id = "agent-" + str(uuid.uuid4())[:8]
    
    # We will send a structured JSON payload with 'tools'
    payload_obj = {
        "messages": [{"role": "user", "content": "Can you read the file /etc/os-release?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file on the host operating system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The absolute path of the file to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        }]
    }

    print("Submitting structured INFER operation...")
    res = transport.submit(
        agent_id=agent_id,
        instance_id="inst-123",
        run_id="run-123",
        operation_id="op-readfile",
        kind_int=1, # OPERATION_KIND_INFER
        payload=json.dumps(payload_obj).encode("utf-8"),
    )
    
    print("\nResult from Kernel:")
    outcome_str = res.get("local_outcome", "{}").strip()
    print(f"RAW: {repr(outcome_str)}")

if __name__ == "__main__":
    main()
