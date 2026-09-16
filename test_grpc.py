import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))

from vectros_sdk.transport.grpc_transport import GrpcTransport

def main():
    print("Connecting to [::1]:50051...")
    # Increase timeout to 300 seconds to allow Ollama time to load the model into GPU memory
    transport = GrpcTransport(address="[::1]:50051", timeout=300.0)
    
    print("Checking capabilities...")
    caps = transport.describe_capabilities()
    print(f"Capabilities: {caps}")

    print("Submitting INFER operation to Rust Kernel...")
    res = transport.submit(
        agent_id="agent-123",
        instance_id="inst-123",
        run_id="run-123",
        operation_id="op-123",
        kind_int=1, # OPERATION_KIND_INFER
        payload=b"Hello, AIOS Kernel! What is 2+2?",
    )
    print(f"Result: {json.dumps(res, indent=2)}")

if __name__ == "__main__":
    main()
