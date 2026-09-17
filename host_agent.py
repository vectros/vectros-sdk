# SEC.15 / SEC.1-class warning — READ BEFORE RUNNING:
#
# This script is an UNSANDBOXED, UNAUDITED PROTOTYPE, not part of the
# packaged `vectros-sdk` distribution (it lives at the repo root, outside
# the `vectros_sdk` package `pyproject.toml` actually ships) and not run by
# any test suite or CI job. `execute_tool("run_command", ...)` below calls
# `subprocess.run(cmd, shell=True)` on whatever command the connected LLM
# backend decides to emit, with the operator's full host privileges and no
# admission check of any kind — a model proposing a destructive command
# will simply run it. It also depends on `GrpcTransport`, which talks to
# `aiosd::grpc_server` (Path B: no `aios-core` admission, no accounting, no
# audit trail — see CLAUDE.md and docs/architecture.md SS1-SS2).
#
# Kept only until the real, sandboxed AIOS Terminal (plan.md Phase 12a,
# TERM.6's proposal+approval path) replaces it — do not extend this script,
# do not wire it into anything else, and never run it against a host or
# network you care about.

import sys
import os
import json
import subprocess
sys.path.insert(0, os.path.dirname(__file__))

from vectros_sdk.transport.grpc_transport import GrpcTransport

def execute_tool(name, args):
    if name == "run_command":
        cmd = args.get("command")
        print(f"\n[Host Agent] Executing command: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return str(e)
    return "Unknown tool"

def main():
    print("Connecting to AIOS Kernel at [::1]:50051...")
    transport = GrpcTransport(address="[::1]:50051", timeout=300.0)
    
    messages = [{"role": "system", "content": "You are a systems agent running on a Linux host. Use the run_command tool to gather information."}]
    messages.append({"role": "user", "content": "What Linux distribution and kernel version is this host running? Run commands to find out."})
    
    tools = [{
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a bash command on the host system",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"]
            }
        }
    }]

    for step in range(3): # Max 3 steps
        payload_obj = {"messages": messages, "tools": tools}
        
        print(f"\n--- Step {step+1} ---")
        print("Submitting INFER operation to Kernel...")
        res = transport.submit(
            agent_id="host-agent",
            instance_id="inst-1",
            run_id="run-1",
            operation_id=f"op-step-{step}",
            kind_int=1, # OPERATION_KIND_INFER
            payload=json.dumps(payload_obj).encode("utf-8"),
        )
        
        outcome_str = res.get("local_outcome", "{}").strip()
        
        # Robustly extract JSON if there's conversational wrapper or markdown
        start_idx, end_idx = -1, -1
        for i, c in enumerate(outcome_str):
            if c in '{[':
                start_idx = i; break
        for i in range(len(outcome_str) - 1, -1, -1):
            if outcome_str[i] in '}]':
                end_idx = i; break
                
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            outcome_str = outcome_str[start_idx:end_idx+1]
        
        try:
            message = json.loads(outcome_str)
            messages.append(message) # Append assistant message to history
            
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    func_name = tc.get('function') if isinstance(tc.get("function"), str) else tc['function']['name']
                    args_dict = tc.get('args') if isinstance(tc.get("function"), str) else json.loads(tc['function']['arguments'])
                    
                    tool_result = execute_tool(func_name, args_dict)
                    print(f"[Host Agent] Tool Result:\n{tool_result.strip()}")
                    
                    # Append tool result to history (handling both strict OpenAI and relaxed Ollama schemas)
                    tc_id = tc.get('id', 'call_1')
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": func_name,
                        "content": tool_result
                    })
            else:
                print(f"\n[Host Agent] Final Answer:\n{message.get('content', '')}")
                break
                
        except json.JSONDecodeError:
            print("Failed to parse outcome from Kernel.")
            print(outcome_str)
            break

if __name__ == "__main__":
    main()
