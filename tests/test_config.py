import pytest

from vectros import Agent, DeepAgent
from vectros.config import agent_settings


def write_config(tmp_path, monkeypatch, text):
    (tmp_path / "vectros.toml").write_text(text)
    monkeypatch.chdir(tmp_path)


def test_agent_takes_its_table_from_vectros_toml(fake, tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, """
[agents.reviewer]
model = "anthropic/claude-opus-5-5"
system = "Be brief."
session = "demo"
max_steps = 6
timeout = 90.5
tool_protocol = "json"
trace = true

[agents.other]
model = "other-model"
""")
    agent = Agent("reviewer", kernel=fake())
    assert agent.model == "anthropic/claude-opus-5-5"
    assert agent.system == "Be brief."
    assert agent.session == "demo"
    assert agent.max_steps == 6
    assert agent.timeout_ms == 90500
    assert agent.tool_protocol == "json"
    assert agent.trace is True


def test_defaults_without_a_config_file(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = Agent("plain", kernel=fake())
    assert (agent.model, agent.max_steps, agent.timeout_ms, agent.tool_protocol) == \
        (None, 10, 120_000, "native")
    assert DeepAgent("deep", kernel=fake()).max_steps == 50


def test_config_is_found_in_a_parent_directory(fake, tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, '[agents.a]\nmodel = "m-parent"\n')
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    assert Agent("a", kernel=fake()).model == "m-parent"


def test_vectros_config_names_the_file(fake, tmp_path, monkeypatch):
    path = tmp_path / "elsewhere.toml"
    path.write_text('[agents.a]\nmodel = "m-explicit"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VECTROS_CONFIG", str(path))
    assert Agent("a", kernel=fake()).model == "m-explicit"
    monkeypatch.setenv("VECTROS_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(ValueError, match="not a file"):
        Agent("a", kernel=fake())


def test_precedence_is_argument_then_environment_then_file(fake, tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, '[agents.a]\nmodel = "from-file"\nmax_steps = 3\n')
    monkeypatch.setenv("VECTROS_MODEL", "from-env")
    monkeypatch.setenv("VECTROS_TIMEOUT", "30")
    agent = Agent("a", kernel=fake())
    assert (agent.model, agent.max_steps, agent.timeout_ms) == ("from-env", 3, 30_000)
    assert Agent("a", model="from-arg", kernel=fake()).model == "from-arg"


def test_deep_agent_uses_configured_system_and_max_steps(fake, tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch,
                 '[agents.planner]\nsystem = "You plan trips."\nmax_steps = 7\n')
    agent = DeepAgent("planner", kernel=fake())
    assert agent.system.startswith("You plan trips.")
    assert agent.max_steps == 7


@pytest.mark.parametrize("text, message", [
    ('[agents.a]\nmodle = "x"\n', "unknown setting agents.a.modle"),
    ("[agents.a]\nmax_steps = true\n", "max_steps for agent 'a' must be int"),
    ("[agents.a]\nmax_steps = 0\n", "must be above 0"),
    ('[agents.a]\ntimeout = "slow"\n', "timeout for agent 'a' must be int or float"),
    ('[agents.a]\ntool_protocol = "xml"\n', "tool_protocol"),
    ('agents = 1\n', "agents"),
    ("[agents\n", "vectros.toml"),
])
def test_bad_settings_are_rejected(fake, tmp_path, monkeypatch, text, message):
    write_config(tmp_path, monkeypatch, text)
    with pytest.raises(ValueError, match=message):
        Agent("a", kernel=fake())


def test_bad_environment_values_are_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VECTROS_MAX_STEPS", "many")
    with pytest.raises(ValueError, match="VECTROS_MAX_STEPS='many' is not a number"):
        agent_settings("a")
    monkeypatch.setenv("VECTROS_MAX_STEPS", "-1")
    with pytest.raises(ValueError, match="must be above 0"):
        agent_settings("a")


def test_choose_model_takes_the_first_allowed_preference(tmp_path):
    from vectros import choose_model, is_local
    allow = tmp_path / "allowed"
    allow.write_text("gemma4:e4b,nvidia/meta/llama\n")
    path = str(allow)
    assert choose_model("anthropic/claude", "nvidia/meta/llama", "gemma4:e4b", path=path) == "nvidia/meta/llama"
    assert choose_model("anthropic/claude", None, path=path) == "gemma4:e4b"
    allow.write_text("*")
    assert choose_model("anthropic/claude", path=path) == "anthropic/claude"
    assert choose_model("x", path=str(tmp_path / "missing")) == "x"
    assert is_local("gemma4:e4b") and is_local("openai-compatible/x") and not is_local("nvidia/a/b")
