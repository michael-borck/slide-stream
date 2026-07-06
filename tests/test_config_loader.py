"""Tests for configuration loading and validation."""

import pytest

from slide_stream.config_loader import (
    ConfigurationError,
    create_example_config,
    expand_env_vars,
    find_config_file,
    load_config,
    merge_configs,
    save_example_config,
    validate_config,
)


@pytest.fixture(autouse=True)
def _no_home_config(monkeypatch):
    """Isolate tests from a real ~/.slidestream.yaml on the dev machine."""
    monkeypatch.setattr(
        "slide_stream.config_loader.find_home_config", lambda: None
    )


# --- expand_env_vars --------------------------------------------------------


def test_expand_env_vars_substitutes_known_var(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret123")
    result = expand_env_vars({"key": "${MY_TOKEN}"})
    assert result == {"key": "secret123"}


def test_expand_env_vars_missing_var_becomes_empty(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    assert expand_env_vars("${NOPE}") == ""


def test_expand_env_vars_recurses_into_lists_and_dicts(monkeypatch):
    monkeypatch.setenv("A", "1")
    data = {"outer": [{"inner": "${A}"}], "plain": "literal"}
    assert expand_env_vars(data) == {"outer": [{"inner": "1"}], "plain": "literal"}


def test_expand_env_vars_leaves_non_template_strings(monkeypatch):
    assert expand_env_vars("just a string") == "just a string"


# --- merge_configs ----------------------------------------------------------


def test_merge_configs_deep_merges_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20, "z": 30}}
    result = merge_configs(base, override)
    assert result == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}


def test_merge_configs_does_not_mutate_base():
    base = {"a": {"x": 1}}
    merge_configs(base, {"a": {"x": 99}})
    assert base == {"a": {"x": 1}}


# --- validate_config --------------------------------------------------------


def test_validate_config_accepts_defaults():
    # Should not raise.
    validate_config(
        {
            "providers": {"llm": {}, "images": {}, "tts": {}},
            "settings": {"video": {"resolution": [1920, 1080]}},
        }
    )


def test_validate_config_missing_section_raises():
    with pytest.raises(ConfigurationError, match="Missing required section"):
        validate_config({"providers": {}})


def test_validate_config_missing_provider_raises():
    with pytest.raises(ConfigurationError, match="Missing provider configuration"):
        validate_config(
            {
                "providers": {"llm": {}, "images": {}},  # no tts
                "settings": {"video": {"resolution": [1, 2]}},
            }
        )


def test_validate_config_bad_resolution_raises():
    with pytest.raises(ConfigurationError, match="resolution"):
        validate_config(
            {
                "providers": {"llm": {}, "images": {}, "tts": {}},
                "settings": {"video": {"resolution": [1920]}},
            }
        )


# --- load_config ------------------------------------------------------------


def test_load_config_returns_defaults_when_no_file(monkeypatch, tmp_path):
    # Run in an empty dir with no home config so no layer applies.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "slide_stream.config_loader.find_home_config", lambda: None
    )
    config = load_config()
    assert config["providers"]["tts"]["provider"] == "gtts"
    assert config["settings"]["video"]["fps"] == 24


def test_load_config_explicit_missing_path_raises():
    with pytest.raises(ConfigurationError, match="not found"):
        load_config("/no/such/config.yaml")


def test_load_config_merges_user_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "slidestream.yaml"
    cfg.write_text(
        "providers:\n"
        "  tts:\n"
        "    provider: elevenlabs\n"
        "settings:\n"
        "  cleanup: false\n"
    )
    config = load_config(str(cfg))
    # Overridden values win...
    assert config["providers"]["tts"]["provider"] == "elevenlabs"
    assert config["settings"]["cleanup"] is False
    # ...while untouched defaults remain.
    assert config["settings"]["video"]["fps"] == 24


def test_load_config_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("providers: [unclosed\n")
    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_config(str(bad))


# --- find_config_file -------------------------------------------------------


def test_find_config_file_locates_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slidestream.yaml").write_text("providers: {}\n")
    found = find_config_file()
    assert found is not None
    assert found.name == "slidestream.yaml"


# --- example config ---------------------------------------------------------


def test_create_example_config_is_valid_yaml():
    import yaml

    parsed = yaml.safe_load(create_example_config())
    assert "providers" in parsed
    assert "settings" in parsed


def test_save_example_config_writes_file(tmp_path):
    out = tmp_path / "example.yaml"
    save_example_config(str(out))
    assert out.exists()
    assert "providers" in out.read_text()


# --- layered config (home + project) ----------------------------------------


def test_home_config_layers_under_project(tmp_path, monkeypatch):
    """Personal home config supplies a server URL; the project config only
    overrides deck-specific settings; both survive the merge."""
    home = tmp_path / "home.yaml"
    home.write_text(
        "providers:\n"
        "  tts:\n"
        "    provider: chatterbox\n"
        "    base_url: https://voice.example.org\n"
    )
    monkeypatch.setattr(
        "slide_stream.config_loader.find_home_config", lambda: home
    )
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    (project / "slidestream.yaml").write_text(
        "providers:\n"
        "  tts:\n"
        "    voice: Michael.wav\n"
        "settings:\n"
        "  cleanup: false\n"
    )

    config = load_config()

    # Home layer supplies the server; project layer supplies the voice.
    assert config["providers"]["tts"]["base_url"] == "https://voice.example.org"
    assert config["providers"]["tts"]["provider"] == "chatterbox"
    assert config["providers"]["tts"]["voice"] == "Michael.wav"
    assert config["settings"]["cleanup"] is False


def test_project_config_overrides_home(tmp_path, monkeypatch):
    """When both set the same key, the project layer wins."""
    home = tmp_path / "home.yaml"
    home.write_text("providers:\n  tts:\n    voice: HomeVoice.wav\n")
    monkeypatch.setattr(
        "slide_stream.config_loader.find_home_config", lambda: home
    )
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    (project / "slidestream.yaml").write_text(
        "providers:\n  tts:\n    voice: ProjectVoice.wav\n"
    )

    config = load_config()
    assert config["providers"]["tts"]["voice"] == "ProjectVoice.wav"


def test_explicit_config_still_layers_over_home(tmp_path, monkeypatch):
    """An explicit --config replaces auto-discovery but keeps the home layer."""
    home = tmp_path / "home.yaml"
    home.write_text("providers:\n  tts:\n    base_url: https://voice.example.org\n")
    monkeypatch.setattr(
        "slide_stream.config_loader.find_home_config", lambda: home
    )
    monkeypatch.chdir(tmp_path)
    explicit = tmp_path / "deck.yaml"
    explicit.write_text("providers:\n  tts:\n    voice: Michael.wav\n")

    config = load_config(str(explicit))
    assert config["providers"]["tts"]["base_url"] == "https://voice.example.org"
    assert config["providers"]["tts"]["voice"] == "Michael.wav"


def test_home_config_alone_applies(tmp_path, monkeypatch):
    """Home config applies even with no project config present."""
    home = tmp_path / "home.yaml"
    home.write_text("settings:\n  strict: true\n")
    monkeypatch.setattr(
        "slide_stream.config_loader.find_home_config", lambda: home
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    config = load_config()
    assert config["settings"]["strict"] is True
