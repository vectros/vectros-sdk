"""
Real-kernel translation layer (M2/SDK.3).

Translates the SDK's high-level `Query` models (`LLMQuery`, `MemoryQuery`,
`StorageQuery`, `ToolQuery`) into real `KernelRequest` calls against a
standing AIOS kernel server (`aiosctl serve-kernel`), and translates the real
`KernelValue` responses back into the flat dicts each subsystem's
`_parse_*_response` function already accepts from the old HTTP mock path —
so none of the ~15 high-level functions in `llm/api.py`, `memory/api.py`,
`storage/api.py`, `tool/api.py`, or their pydantic response models, need to
change.

Not every operation the mock/HTTP API allowed has a real-kernel equivalent.
Those raise `RealBackendUnsupported` rather than fabricating a result —
never a silent mock fallback for a real-looking call:

- Memory `search` / `create_agentic` — no vector/semantic memory in the
  kernel (ARCH.6: deliberately out of scope, not merely unimplemented).
- Storage `mount` / `create_dir` / `retrieve_file` / `rollback_file` /
  `share_file` — the real Storage Manager is a flat `(collection, object)`
  key-value store with exact-key get/put/delete/list, not a hierarchical
  filesystem. `retrieve_file` in particular is natural-language *search*
  over file content, which the real backend cannot do at all — this is a
  materially bigger gap than it first appears. Only `create_file` and
  `write_file` (both exact-key writes) have a real equivalent.
- LLM `tool_use` / `operate_file` — `execution.rs`'s real-backend path
  always returns empty `tool_suggestions` today (M1 left real-backend
  tool-call parsing as a follow-up). Claiming to support `tool_use` would
  silently look like "the model chose not to call a tool" when the real
  reason is "this isn't wired up yet" — so it is refused, not faked.
- Every Post operation (`send`, `receive`, `broadcast`, `publish`,
  `subscribe`) — the kernel's real primitive is `IpcSend`/`IpcReceive`
  between two *known* agents with an existing grant, not topic-based
  pub/sub. There is no translation, not even a partial one.
- Tool calls for anything except the one pre-registered fixture tool
  (`res_tool_uppercase_<agent>`, an uppercase text transform) — the
  kernel's `ToolEntrypoint` is a closed set of reviewed fixtures, not a
  bring-your-own-Python-callable registry like the SDK's client-side tool
  registry (`vectros_sdk.tool.core.registry`).

Conversation model for `llm_chat` (a real, recorded design decision, not an
oversight): the real kernel has no "stateless, full-history-per-call"
primitive — a `Context` is a server-side object with its own revision
counter, and creating one twice is a `Duplicate` error. Recreating a fresh
context per call would need a fresh granted resource per call, which the
current single-tenant bootstrap does not (and should not, for a fixed grant
set) support. So: **one persistent context per agent for the process
lifetime; each call appends only the newest message** (`messages[-1]`) and
generates against the accumulated conversation. Earlier entries in the
caller's `messages` list are assumed to already be part of that accumulated
context from prior calls — passing a full OpenAI-style growing history every
call will duplicate earlier turns. This is documented here, not hidden.

One-fixed-resource-per-agent applies to Memory and Storage too (a real,
confirmed constraint of `kernel_server.rs::tenant_permissions`, not
something this module invents): `memory.get/put/delete` are granted on
exactly one resource, `res_memory_<suffix>`, and `storage.get/put/delete`
are checked against the request's *object* id (not its collection), which
is granted only for `res_storage_<suffix>` — the same id the one storage
collection itself uses. Concretely:

- Memory: every agent has exactly one real memory slot. `create` writes it;
  `get`/`update`/`delete` against any *other* `memory_id` raise
  `RealBackendUnsupported` rather than a confusing server-side
  `permission_denied` — there is nowhere else for them to point yet. A
  second `create` without an intervening `delete` fails with a real
  revision-conflict error from the server (not swallowed into a fake
  success), since it is genuinely overwriting the one slot from a
  non-zero revision.
- Storage: every agent has exactly one real storage slot, addressed by
  `file_path` in name only — `create_file`/`write_file` always write the
  same underlying object regardless of which path is given. This is a
  materially bigger limitation than "no directories": there is currently
  no way to address more than one real file per agent at all. Multi-object
  storage needs per-object grants that `tenant_permissions` does not issue
  today (a real follow-up, not scoped to SDK.3).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from vectros_sdk.core.models import Query
from vectros_sdk.transport.execution_protocol import (
    ExecutionProtocolClient,
    context_append,
    context_create,
    context_prepare,
    memory_delete,
    memory_get,
    memory_put,
    model_generate,
    storage_put,
    tool_invoke,
)

# Real generations always cost real tokens; there is no per-call budget knob
# in LLMQuery today, so this default is a documented limitation, not a
# silent guess. Headroom (`output_bytes`) must satisfy execution.rs's
# `max_output_tokens * 16 <= reserved_output_bytes` check.
_DEFAULT_MAX_OUTPUT_TOKENS = 1024
_DEFAULT_OUTPUT_BYTES = _DEFAULT_MAX_OUTPUT_TOKENS * 16 + 512
# A real generation up to the full _DEFAULT_MAX_OUTPUT_TOKENS budget
# (including a "thinking" model spending most of it on internal reasoning
# before any final content, see llm_backend.rs's own real fallback for that
# case) has been observed live taking 45+ seconds -- ExecutionProtocolClient
# .execute()'s own 30_000ms default (this module's other calls are all fast,
# local, non-generating kernel operations, so that default stays right for
# them) is not enough headroom for the one call here that can legitimately
# run long. Matches OllamaConfig::DEFAULT_TIMEOUT (llm_backend.rs) so the
# client-side poll budget is never shorter than what the server's own real
# network call to Ollama is itself allowed to take.
_GENERATION_DEADLINE_MS = 120_000
_DEFAULT_WINDOW_BYTES = _DEFAULT_OUTPUT_BYTES * 4

# Tracks each agent's context revision (and, separately, its one storage
# object's version) for this process only. Two processes driving the same
# agent_name against the same server will conflict — this is a real
# limitation of the current single-tenant bootstrap, not a bug in this
# module.
_context_revisions: Dict[str, int] = {}
_storage_versions: Dict[str, int] = {}

# Friendly tool name -> the fixture tool `kernel_server.rs` registers.
# Extend this only when a new real tool is actually registered server-side;
# never invent a mapping the server does not back.
_KNOWN_TOOLS = {"uppercase"}


class RealBackendUnsupported(Exception):
    """Raised when a query has no equivalent in the real AIOS protocol.

    Never silently falls back to a mock or fabricates a response.
    """


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _agent_suffix(query: Query) -> str:
    agent_name = getattr(query, "agent_name", None) or "default"
    return str(agent_name)


def model_resource_id(model_name: str) -> str:
    """Mirrors `aiosd::model_resource_id` exactly — model names commonly
    contain `:` (e.g. `gemma4:e4b`), which the kernel's `ResourceId` rejects.
    """
    return f"res_model_{model_name.replace(':', '-')}"


def uppercase_tool_resource_id(agent_suffix: str) -> str:
    """Mirrors `aiosd::uppercase_tool_resource_id`."""
    return f"res_tool_uppercase_{agent_suffix}"


def execute_real(query: Query, socket_path: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Dispatches `query` against a real, standing kernel server.

    Returns a flat dict shaped like the subsystem's expected raw response, so
    the existing `_parse_*_response` functions need no changes.

    Raises `RealBackendUnsupported` for any operation with no real
    equivalent. Raises `vectros_sdk.transport.execution_protocol.
    ExecutionProtocolError` for a genuine server rejection (permission
    denied, malformed request, etc.) — never swallowed into a fake success.
    """
    query_class = getattr(query, "query_class", None)
    with ExecutionProtocolClient(socket_path, timeout=timeout) as client:
        if query_class == "memory":
            return _memory(query, client)
        if query_class == "storage":
            return _storage(query, client)
        if query_class == "tool":
            return _tool(query, client)
        if query_class == "llm":
            return _llm(query, client)
        if query_class == "post":
            raise RealBackendUnsupported(
                "post/* has no equivalent in the real AIOS protocol: the "
                "kernel's IPC primitive is point-to-point between two known "
                "agents with an existing grant, not topic-based pub/sub."
            )
        raise RealBackendUnsupported(f"unknown query_class: {query_class!r}")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _real_memory_resource(suffix: str) -> str:
    """The one memory resource `tenant_permissions` grants for this agent."""
    return f"res_memory_{suffix}"


def _memory(query: Any, client: ExecutionProtocolClient) -> Dict[str, Any]:
    suffix = _agent_suffix(query)
    key = _real_memory_resource(suffix)
    action = query.action_type

    if action in ("get", "update", "delete") and query.memory_id and query.memory_id != key:
        raise RealBackendUnsupported(
            f"memory.{action} against {query.memory_id!r} has no real equivalent: "
            f"this agent's only granted memory resource is {key!r} (one fixed slot "
            "per agent — tenant_permissions does not issue per-key grants yet)."
        )

    if action == "create":
        value = client.execute(_new_id("req_mem_put"), memory_put(key, query.content or ""))
        entry = value["Data"]["MemoryEntry"]["entry"]
        return {
            "success": True,
            "finished": True,
            "memory_id": entry["key"],
            "content": query.content,
            "metadata": query.metadata,
        }

    if action == "get":
        value = client.execute(_new_id("req_mem_get"), memory_get(key))
        entry = value["Data"]["MemoryEntry"]["entry"]
        return {
            "success": True,
            "finished": True,
            "memory_id": entry["key"],
            "content": (entry["value"] or {}).get("Text"),
            "metadata": query.metadata,
        }

    if action == "update":
        current = client.execute(_new_id("req_mem_read"), memory_get(key))
        revision = current["Data"]["MemoryEntry"]["entry"]["revision"]
        value = client.execute(
            _new_id("req_mem_put"),
            memory_put(key, query.content or "", expected_revision=revision),
        )
        entry = value["Data"]["MemoryEntry"]["entry"]
        return {
            "success": True,
            "finished": True,
            "memory_id": entry["key"],
            "content": query.content,
            "metadata": query.metadata,
        }

    if action == "delete":
        current = client.execute(_new_id("req_mem_read"), memory_get(key))
        revision = current["Data"]["MemoryEntry"]["entry"]["revision"]
        client.execute(_new_id("req_mem_delete"), memory_delete(key, expected_revision=revision))
        return {"success": True, "finished": True, "memory_id": key}

    if action in ("search", "create_agentic"):
        raise RealBackendUnsupported(
            f"memory.{action} has no equivalent in the real AIOS kernel "
            "(no vector/semantic memory — ARCH.6 marks this out of scope)."
        )
    raise RealBackendUnsupported(f"unknown memory action_type: {action!r}")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _first_param(query: Any) -> Dict[str, Any]:
    params = query.params
    if isinstance(params, list) and params:
        return params[0]
    if isinstance(params, dict):
        return params
    return {}


def _storage(query: Any, client: ExecutionProtocolClient) -> Dict[str, Any]:
    suffix = _agent_suffix(query)
    # `tenant_permissions` checks storage.put/get/delete against the
    # request's *object* id, granted only for `res_storage_<suffix>` — the
    # same id the sole collection itself uses (see this module's docstring).
    # There is exactly one writable object per agent; `file_path` cannot
    # address a second one yet.
    collection = f"res_storage_{suffix}"
    obj = collection
    operation = query.operation_type
    param = _first_param(query)

    if operation in ("create_file", "write_file"):
        file_path = param.get("file_path")
        if not file_path:
            raise RealBackendUnsupported(f"storage.{operation} requires file_path")
        content = param.get("content", "")
        version = _storage_versions.get(suffix, 0)
        value = client.execute(
            _new_id("req_storage_put"),
            storage_put(collection, obj, content.encode("utf-8"), expected_version=version),
        )
        metadata = value["Data"]["StorageMetadata"]["metadata"]
        _storage_versions[suffix] = metadata["version"]
        return {"finished": True, "response_message": f"wrote {file_path} (single real slot)"}

    if operation in ("create_dir", "mount", "share_file", "rollback_file", "retrieve_file"):
        detail = (
            "natural-language search over file content, which the real "
            "backend cannot do at all"
            if operation == "retrieve_file"
            else "directories/mounts/sharing/version-rollback do not exist "
            "in the real flat (collection, object) Storage Manager"
        )
        raise RealBackendUnsupported(f"storage.{operation} has no real equivalent: {detail}.")

    raise RealBackendUnsupported(f"unknown storage operation_type: {operation!r}")


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _tool(query: Any, client: ExecutionProtocolClient) -> Dict[str, Any]:
    suffix = _agent_suffix(query)
    calls: List[Dict[str, Any]] = query.tool_calls or []
    if not calls:
        raise RealBackendUnsupported("tool query has no tool_calls")
    if len(calls) > 1:
        raise RealBackendUnsupported(
            "batched tool_calls are not yet supported against the real "
            "kernel; only the first call is ever executed by the mock path "
            "either — pass one call at a time."
        )
    call = calls[0]
    name = call.get("name")
    if name not in _KNOWN_TOOLS:
        raise RealBackendUnsupported(
            f"tool {name!r} is not registered with the real kernel; only "
            f"{sorted(_KNOWN_TOOLS)} exist (a closed, reviewed fixture set, "
            "not a bring-your-own-callable registry)."
        )
    parameters = call.get("parameters", {})
    input_text = parameters.get("input")
    if input_text is None:
        raise RealBackendUnsupported("the 'uppercase' tool requires parameters={'input': str}")

    tool_resource = uppercase_tool_resource_id(suffix)
    value = client.execute(
        _new_id("req_tool"), tool_invoke(tool_resource, {"input": str(input_text)})
    )
    result_pairs = value.get("ToolResult", [])
    output = next((v.get("String") for k, v in result_pairs if k == "output"), None)
    return {"finished": True, "response_message": output}


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def _llm(query: Any, client: ExecutionProtocolClient) -> Dict[str, Any]:
    if query.action_type in ("tool_use", "operate_file"):
        raise RealBackendUnsupported(
            f"llm.{query.action_type} is not supported against the real kernel yet: "
            "real-backend tool-call parsing is a tracked follow-up (M1 left "
            "`suggestions` empty for every real generation); claiming support "
            "here would look like 'the model chose not to call a tool' when "
            "the real reason is that the plumbing does not exist."
        )
    if not query.messages:
        raise RealBackendUnsupported("llm.chat requires at least one message")
    if not query.llms:
        raise RealBackendUnsupported(
            "llm.chat against the real kernel requires llms=[{'name': "
            "'<model>'}] naming the exact model the server was started with "
            "(aiosctl serve-kernel --ollama-model <name>) — there is no "
            "default, and choosing the wrong name is a real rejection, not "
            "a silent fallback."
        )

    suffix = _agent_suffix(query)
    context_id = f"context_context_{suffix}"
    if suffix not in _context_revisions:
        client.execute(_new_id("req_ctx_create"), context_create(context_id))
        _context_revisions[suffix] = 0
    revision = _context_revisions[suffix]

    last_message = query.messages[-1]
    role = str(last_message.get("role", "user")).capitalize()
    text = str(last_message.get("content", ""))
    append_value = client.execute(
        _new_id("req_ctx_append"), context_append(context_id, revision, role, text)
    )
    new_revision = append_value["Revision"]
    _context_revisions[suffix] = new_revision

    model = model_resource_id(query.llms[0]["name"])
    prepare_id = _new_id("req_ctx_prepare")
    prepare_value = client.execute(
        prepare_id,
        context_prepare(
            context_id, new_revision, model, _DEFAULT_OUTPUT_BYTES, _DEFAULT_WINDOW_BYTES
        ),
    )
    input_bytes = prepare_value["Prepared"]["input_bytes"]

    generate_value = client.execute(
        _new_id("req_gen"),
        model_generate(prepare_id, model, input_bytes, _DEFAULT_MAX_OUTPUT_TOKENS),
        deadline_ms=_GENERATION_DEADLINE_MS,
    )
    generated = generate_value["Generated"]
    return {
        "response_message": "".join(generated["tokens"]),
        "tool_calls": None,
        "finished": True,
    }
