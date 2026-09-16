import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))

from vectros_sdk.transport.grpc_transport import GrpcTransport

def main():
    print("Connecting to [::1]:50051...")
    transport = GrpcTransport(address="[::1]:50051", timeout=300.0)
    
    # We will send a structured JSON payload with 'tools'
    payload_obj = {
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
    }

    print("Submitting structured INFER operation...")
    res = transport.submit(
        agent_id="agent-123",
        instance_id="inst-123",
        run_id="run-123",
        operation_id="op-weather",
        kind_int=1, # OPERATION_KIND_INFER
        payload=json.dumps(payload_obj).encode("utf-8"),
    )
    
    print("\nResult from Kernel:")
    # The kernel returns the entire message object as a JSON string in local_outcome
    outcome_str = res.get("local_outcome", "{}").strip()
    
    # Robustly extract JSON if there's conversational wrapper or markdown
    start_idx = -1
    end_idx = -1
    for i, c in enumerate(outcome_str):
        if c in '{[':
            start_idx = i
            break
    for i in range(len(outcome_str) - 1, -1, -1):
        if outcome_str[i] in '}]':
            end_idx = i
            break
            
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        outcome_str = outcome_str[start_idx:end_idx+1]
        
    print(f"RAW EXTRACTED JSON: {repr(outcome_str)}")
    
    try:
        message = json.loads(outcome_str)
        print(json.dumps(message, indent=2))
        
        if "tool_calls" in message and message["tool_calls"]:
            print("\nSUCCESS: The LLM invoked a tool!")
            for tc in message["tool_calls"]:
                print(f"Tool Call ID: {tc.get('id', 'N/A')}")
                if isinstance(tc.get("function"), str):
                    # Ollama's direct tool schema sometimes outputs raw names
                    print(f"Function: {tc.get('function')}")
                    print(f"Arguments: {tc.get('args', {})}")
                else:
                    # OpenAI's nested schema
                    print(f"Function: {tc['function']['name']}")
                    print(f"Arguments: {tc['function']['arguments']}")
        else:
            print("\nNo tool calls found. The LLM generated text instead.")
            
    except json.JSONDecodeError:
        print("Failed to parse local_outcome as JSON:")
        print(outcome_str)

if __name__ == "__main__":
    main()
