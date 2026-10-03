import pytest

from vectros import cli


def test_init_creates_project_and_refuses_overwrite(tmp_path, capsys):
    project = tmp_path / "helper"
    assert cli.main(["init", str(project)]) == 0
    config = cli.load_config(project)
    assert config["agent"]["name"] == "helper"
    assert 'Agent("helper"' in (project / "agent.py").read_text()
    assert cli.main(["init", str(project)]) == 1
    assert "already exists" in capsys.readouterr().err


def test_load_config_rejects_entry_outside_project(tmp_path):
    (tmp_path / "vectros.toml").write_text('[agent]\nname = "x"\nentry = "../x.py"\n')
    with pytest.raises(cli.CliError, match="not a file in the project"):
        cli.load_config(tmp_path)


@pytest.mark.parametrize("host", ["-oProxyCommand=x", "a b", "u@h;rm"])
def test_host_validation(tmp_path, host):
    cli.main(["init", str(tmp_path / "p")])
    with pytest.raises(cli.CliError, match="invalid host"):
        cli._host(type("A", (), {"host": host})(), {})


def test_deploy_runs_expected_commands(tmp_path, monkeypatch):
    project = tmp_path / "bot"
    cli.main(["init", str(project)])
    calls = []
    monkeypatch.setattr(cli, "_run", lambda command, **kw: calls.append((command, kw)))
    assert cli.main(["deploy", "me@box", "-C", str(project)]) == 0
    assert [c[0][0] for c in calls] == ["ssh", "rsync", "ssh", "ssh"]
    rsync = calls[1][0]
    assert rsync[-1] == "me@box:.local/share/vectros/agents/bot/"
    assert "--delete" in rsync and ".venv" in rsync
    unit = calls[2][1]["input"].decode()
    assert "ExecStart=%h/.local/share/vectros/agents/%i/.venv/bin/python -m vectros run" in unit
    assert "systemctl --user restart vectros-agent@bot" in calls[3][0][-1]


def test_run_executes_entry(tmp_path, monkeypatch, capsys):
    project = tmp_path / "p"
    project.mkdir()
    (project / "vectros.toml").write_text('[agent]\nname = "p"\nentry = "main.py"\n')
    (project / "main.py").write_text("import sys\nprint('ran', sys.argv[1:])\n")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", "-C", str(project), "a", "b"]) == 0
    assert "ran ['a', 'b']" in capsys.readouterr().out
