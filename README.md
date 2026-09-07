# Vectros SDK (AIOS-Agent SDK)

`vectros-sdk` provides structured interfaces between user device applications and the AIOS kernel.

## Operation Interface

The legacy module APIs use an HTTP `/query` compatibility endpoint. They are
not the AIOS Operation Interface. For the current development daemon, use the
typed Unix-domain `OperationClient` for lifecycle observation:

```python
from vectros_sdk import OperationClient

client = OperationClient("/tmp/aiosd-dev.sock")
view = client.submit(
    agent_id="agent:example", instance_id="instance:example", run_id="run:example",
    operation_id="operation:example", kind="infer", side_effect="read_only",
)
```

This is intentionally a development-only Interface. It rejects tool invocation
and does not carry a Capability Grant or Approval. Do not use it to dispatch
external effects. Production tool dispatch requires the authenticated Operation
Interface tracked by the kernel's `TOOL-006` work item.
