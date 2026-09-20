"""
ASBX.13: the `agent_sandbox` pytest fixture.

Auto-loads for any project that installs this package (registered under the
`pytest11` entry point in `pyproject.toml`) -- no per-repo `conftest.py`
wiring needed, matching "writing an agent test should cost about what a
unit test costs." A test that wants a real backend passes
`@pytest.mark.agent_sandbox(model="gemma4:e4b")`; one that doesn't just
takes the fixture as-is.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from vectros_sdk.testing.agent_sandbox import AgentSandbox


def pytest_configure(config: "pytest.Config") -> None:
    config.addinivalue_line(
        "markers",
        "agent_sandbox(model=None, **kwargs): configure the agent_sandbox fixture "
        "for this test (see vectros_sdk.testing.AgentSandbox's own constructor "
        "for accepted kwargs).",
    )


@pytest.fixture
def agent_sandbox(request: "pytest.FixtureRequest") -> Iterator[AgentSandbox]:
    """A fresh, real `AgentSandbox` per test -- entered and torn down
    automatically. Configure it with
    `@pytest.mark.agent_sandbox(model="gemma4:e4b", ...)`."""
    marker = request.node.get_closest_marker("agent_sandbox")
    kwargs = dict(marker.kwargs) if marker else {}
    with AgentSandbox(**kwargs) as sandbox:
        yield sandbox
