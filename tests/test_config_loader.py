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


def test_expand_env_vars_missing_var_warns_on_stderr(monkeypatch, capsys):
    """A typo'd ${VAR} must not silently become "" — warn, naming the
    variable and the config key path."""
    monkeypatch.delenv("CHATTERBOX_TOKEN_TYPO", raising=False)
    result = expand_env_vars(
        {"providers": {"tts": {"api_key": "${CHATTERBOX_TOKEN_TYPO}"}}}
    )
    assert result == {"providers": {"tts": {"api_key": ""}}}
    err = capsys.readouterr().err
    assert "CHATTERBOX_TOKEN_TYPO" in err
    assert "not set" in err
    assert "providers.tts.api_key" in err
    assert "empty" in err


def test_expand_env_vars_set_var_does_not_warn(monkeypatch, capsys):
    monkeypatch.setenv("PRESENT", "value")
    expand_env_vars({"key": "${PRESENT}"})
    assert capsys.readouterr().err == ""


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


def test_merge_configs_provider_change_resets_sibling_keys():
    """Changing a provider must not inherit the old provider's settings
    (e.g. an elevenlabs voice leaking into a voicebox block)."""
    from slide_stream.config_loader import DEFAULT_CONFIG

    base = merge_configs(
        DEFAULT_CONFIG,
        {"providers": {"tts": {"provider": "elevenlabs", "voice": "rachel"}}},
    )
    result = merge_configs(
        base,
        {"providers": {"tts": {"provider": "voicebox",
                               "base_url": "https://voice.example.org"}}},
    )
    tts = result["providers"]["tts"]
    assert tts["provider"] == "voicebox"
    assert tts["base_url"] == "https://voice.example.org"
    # The elevenlabs voice does not leak; the block restarts from defaults.
    assert tts["voice"] is None
    assert tts["engine"] == "kokoro"


def test_merge_configs_same_provider_inherits_sibling_keys():
    base = {
        "providers": {
            "tts": {"provider": "voicebox", "voice": "Michael.wav",
                    "base_url": None}
        }
    }
    result = merge_configs(
        base,
        {"providers": {"tts": {"provider": "voicebox",
                               "base_url": "https://voice.example.org"}}},
    )
    tts = result["providers"]["tts"]
    assert tts["voice"] == "Michael.wav"
    assert tts["base_url"] == "https://voice.example.org"


def test_merge_configs_null_section_keeps_base():
    """A commented-out section (YAML null) keeps the lower layer's mapping."""
    base = {"providers": {"tts": {"provider": "voicebox"}}}
    result = merge_configs(base, {"providers": None})
    assert result["providers"]["tts"]["provider"] == "voicebox"


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
    assert config["providers"]["tts"]["provider"] == "voicebox"
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


def test_load_config_null_providers_section_keeps_defaults(tmp_path, monkeypatch):
    """`providers:` with all children commented out must not crash — the
    defaults apply instead."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "slidestream.yaml"
    cfg.write_text("providers:\nsettings:\n  cleanup: false\n")
    config = load_config(str(cfg))
    assert config["providers"]["tts"]["provider"] == "voicebox"
    assert config["settings"]["cleanup"] is False


def test_load_config_null_video_section_keeps_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "slidestream.yaml"
    cfg.write_text("settings:\n  video:\n")
    config = load_config(str(cfg))
    assert config["settings"]["video"]["resolution"] == [1920, 1080]


def test_load_config_list_root_raises(tmp_path, monkeypatch):
    """A YAML root that isn't a mapping is a clean error naming the file."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "slidestream.yaml"
    cfg.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigurationError, match="mapping") as excinfo:
        load_config(str(cfg))
    assert "slidestream.yaml" in str(excinfo.value)


def test_load_config_string_root_raises(tmp_path):
    cfg = tmp_path / "oops.yaml"
    cfg.write_text("just a string\n")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_config(str(cfg))


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


def test_save_example_config_is_owner_only(tmp_path):
    """The example config gets filled with API keys — 0600, owner only."""
    out = tmp_path / "example.yaml"
    save_example_config(str(out))
    assert (out.stat().st_mode & 0o777) == 0o600


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


def test_project_provider_change_drops_home_siblings(tmp_path, monkeypatch):
    """When the project switches TTS provider, the home layer's
    provider-specific keys (e.g. an elevenlabs voice) must not leak in."""
    home = tmp_path / "home.yaml"
    home.write_text(
        "providers:\n"
        "  tts:\n"
        "    provider: elevenlabs\n"
        "    voice: rachel\n"
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
        "    provider: voicebox\n"
        "    base_url: https://voice.example.org\n"
    )

    config = load_config()
    tts = config["providers"]["tts"]
    assert tts["provider"] == "voicebox"
    assert tts["base_url"] == "https://voice.example.org"
    assert tts["voice"] is None  # rachel does not leak into voicebox


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
