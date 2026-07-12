"""Tests for contrib/voicebox/sweep_ephemeral_profiles.py.

The sweep deletes voice clones off a server, so the tests lean hard on the
question of what it must *never* touch.
"""

import importlib.util
import sys
import urllib.error
from datetime import datetime, timedelta, timezone
from email.message import Message
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "contrib"
    / "voicebox"
    / "sweep_ephemeral_profiles.py"
)


@pytest.fixture(scope="module")
def sweep_mod():
    spec = importlib.util.spec_from_file_location("sweep_ephemeral_profiles", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profile(name, *, age_minutes=120, voice_type="cloned", pid="p1"):
    created = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    return {
        "id": pid,
        "name": name,
        "voice_type": voice_type,
        # Voicebox serialises datetime.utcnow(): naive, implicitly UTC.
        "created_at": created.replace(tzinfo=None).isoformat(),
    }


EPHEMERAL = "slide-stream-3f2504e0-4f89-41d3-9a0c-0305e82c3301"


def test_only_matches_prefix_plus_uuid(sweep_mod):
    """A hand-made profile must never match, whatever it is called."""
    assert sweep_mod.EPHEMERAL_NAME.match(EPHEMERAL)
    for name in [
        "slide-stream-backup",           # prefix but not a UUID
        "slide-stream-",
        "Lecturer A",
        f"x{EPHEMERAL}",                 # not anchored at the start
        f"{EPHEMERAL}-keep",             # not anchored at the end
        EPHEMERAL.upper(),
    ]:
        assert not sweep_mod.EPHEMERAL_NAME.match(name), name


def test_skips_young_wrong_type_and_named_profiles(sweep_mod, mocker):
    """Only old, cloned, uuid-named profiles are candidates."""
    profiles = [
        _profile(EPHEMERAL, age_minutes=5, pid="young"),
        _profile(EPHEMERAL, voice_type="preset", pid="preset"),
        _profile("Lecturer A", pid="named"),
        _profile(EPHEMERAL, pid="stale"),
    ]
    mocker.patch.object(sweep_mod, "_request", return_value=profiles)

    stale = sweep_mod.find_stale_profiles(
        "https://vb.org", None, timedelta(minutes=60)
    )
    assert [p["id"] for p in stale] == ["stale"]


def test_an_in_flight_run_is_protected_by_the_grace_period(sweep_mod, mocker):
    """A profile created 10 minutes ago belongs to a render still in progress."""
    mocker.patch.object(
        sweep_mod, "_request", return_value=[_profile(EPHEMERAL, age_minutes=10)]
    )
    assert sweep_mod.find_stale_profiles(
        "https://vb.org", None, timedelta(minutes=60)
    ) == []


def test_deletes_generations_before_profile(sweep_mod, mocker):
    """Generations do not cascade, so they must go first, while still listable."""
    def fake_request(method, url, token, timeout=60.0):
        if method == "GET" and url.endswith("/profiles"):
            return [_profile(EPHEMERAL, pid="prof1")]
        if method == "GET" and "/history?" in url:
            return {
                "items": [
                    {"id": "g1", "profile_id": "prof1"},
                    {"id": "g2", "profile_id": "prof1"},
                ],
                "total": 2,
            }
        return None

    request = mocker.patch.object(sweep_mod, "_request", side_effect=fake_request)
    failures = sweep_mod.sweep(
        "https://vb.org", None, timedelta(minutes=60), False, False
    )
    assert failures == 0

    deletes = [c[0][1] for c in request.call_args_list if c[0][0] == "DELETE"]
    assert deletes == [
        "https://vb.org/history/g1",
        "https://vb.org/history/g2",
        "https://vb.org/profiles/prof1",
    ]


def test_dry_run_deletes_nothing(sweep_mod, mocker):
    def fake_request(method, url, token, timeout=60.0):
        if method == "GET" and url.endswith("/profiles"):
            return [_profile(EPHEMERAL, pid="prof1")]
        if method == "GET" and "/history?" in url:
            return {"items": [{"id": "g1", "profile_id": "prof1"}], "total": 1}
        raise AssertionError(f"dry run performed a {method}")

    mocker.patch.object(sweep_mod, "_request", side_effect=fake_request)
    assert sweep_mod.sweep("https://vb.org", None, timedelta(minutes=60), True, False) == 0


def test_keep_generations_leaves_history_alone(sweep_mod, mocker):
    def fake_request(method, url, token, timeout=60.0):
        if method == "GET" and url.endswith("/profiles"):
            return [_profile(EPHEMERAL, pid="prof1")]
        if "/history" in url:
            raise AssertionError("history must not be touched")
        return None

    request = mocker.patch.object(sweep_mod, "_request", side_effect=fake_request)
    assert sweep_mod.sweep("https://vb.org", None, timedelta(minutes=60), False, True) == 0
    deletes = [c[0][1] for c in request.call_args_list if c[0][0] == "DELETE"]
    assert deletes == ["https://vb.org/profiles/prof1"]


def test_history_paginates(sweep_mod, mocker):
    pages = [
        {
            "items": [{"id": f"g{i}", "profile_id": "prof1"} for i in range(100)],
            "total": 150,
        },
        {
            "items": [{"id": f"g{i}", "profile_id": "prof1"} for i in range(100, 150)],
            "total": 150,
        },
    ]
    mocker.patch.object(sweep_mod, "_request", side_effect=pages)
    ids = sweep_mod.generation_ids("https://vb.org", None, "prof1")
    assert len(ids) == 150
    assert ids[0] == "g0" and ids[-1] == "g149"


def test_history_missing_total_pages_until_empty(sweep_mod, mocker):
    """A server that omits 'total' must not stop the listing after one page."""
    pages = [
        {"items": [{"id": f"g{i}", "profile_id": "prof1"} for i in range(100)]},
        {"items": [{"id": "g100", "profile_id": "prof1"}]},
        {"items": []},
    ]
    mocker.patch.object(sweep_mod, "_request", side_effect=pages)
    ids = sweep_mod.generation_ids("https://vb.org", None, "prof1")
    assert len(ids) == 101
    assert ids[-1] == "g100"


def test_history_filter_ignored_by_server_deletes_nothing_foreign(sweep_mod, mocker, capsys):
    """If the server ignores ?profile_id= and returns everyone's history, only
    items whose own profile_id matches may be deleted; items lacking the field
    are skipped and reported, never deleted."""
    page = {
        "items": [
            {"id": "mine", "profile_id": "prof1"},
            {"id": "other", "profile_id": "prof2"},   # another profile's audio
            {"id": "legacy"},                          # no profile_id field
        ],
        "total": 3,
    }
    mocker.patch.object(sweep_mod, "_request", side_effect=[page, {"items": []}])
    ids = sweep_mod.generation_ids("https://vb.org", None, "prof1")
    assert ids == ["mine"]
    captured = capsys.readouterr()
    assert "skipped 1 history item(s)" in captured.err


def test_generation_delete_failure_skips_profile_delete(sweep_mod, mocker):
    """A profile whose generations could not all be removed must survive, so
    the next run can retry — deleting it would orphan the leftovers."""
    def fake_request(method, url, token, timeout=60.0):
        if method == "GET" and url.endswith("/profiles"):
            return [_profile(EPHEMERAL, pid="prof1")]
        if method == "GET" and "/history?" in url:
            return {
                "items": [{"id": "g1", "profile_id": "prof1"}],
                "total": 1,
            }
        if method == "DELETE" and "/history/" in url:
            raise RuntimeError("server hiccup")
        if method == "DELETE" and "/profiles/" in url:
            raise AssertionError("profile must not be deleted")
        return None

    mocker.patch.object(sweep_mod, "_request", side_effect=fake_request)
    failures = sweep_mod.sweep(
        "https://vb.org", None, timedelta(minutes=60), False, False
    )
    assert failures == 1


def test_generation_delete_404_counts_as_gone(sweep_mod, mocker):
    """A 404 on a generation delete means it is already gone — success, and
    the profile delete still proceeds."""
    def fake_request(method, url, token, timeout=60.0):
        if method == "GET" and url.endswith("/profiles"):
            return [_profile(EPHEMERAL, pid="prof1")]
        if method == "GET" and "/history?" in url:
            return {
                "items": [{"id": "g1", "profile_id": "prof1"}],
                "total": 1,
            }
        if method == "DELETE" and "/history/" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        return None

    request = mocker.patch.object(sweep_mod, "_request", side_effect=fake_request)
    failures = sweep_mod.sweep(
        "https://vb.org", None, timedelta(minutes=60), False, False
    )
    assert failures == 0
    deletes = [c[0][1] for c in request.call_args_list if c[0][0] == "DELETE"]
    assert "https://vb.org/profiles/prof1" in deletes


def test_concurrent_sweep_404_is_not_a_failure(sweep_mod, mocker):
    """Another sweep reaping the same profile first is fine, not an error."""
    def fake_request(method, url, token, timeout=60.0):
        if method == "GET" and url.endswith("/profiles"):
            return [_profile(EPHEMERAL, pid="prof1")]
        if method == "GET":
            return {"items": [], "total": 0}
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    mocker.patch.object(sweep_mod, "_request", side_effect=fake_request)
    assert sweep_mod.sweep("https://vb.org", None, timedelta(minutes=60), False, False) == 0


def test_failed_profile_delete_reports_failure(sweep_mod, mocker):
    """Cron must hear about a clone still sitting on the server."""
    def fake_request(method, url, token, timeout=60.0):
        if method == "GET" and url.endswith("/profiles"):
            return [_profile(EPHEMERAL, pid="prof1")]
        if method == "GET":
            return {"items": [], "total": 0}
        raise RuntimeError("server down")

    mocker.patch.object(sweep_mod, "_request", side_effect=fake_request)
    assert sweep_mod.sweep("https://vb.org", None, timedelta(minutes=60), False, False) == 1


def test_main_requires_base_url(sweep_mod, monkeypatch):
    monkeypatch.delenv("VOICEBOX_BASE_URL", raising=False)
    with pytest.raises(SystemExit):
        sweep_mod.main([])


def test_main_returns_nonzero_on_failure(sweep_mod, mocker):
    mocker.patch.object(sweep_mod, "sweep", return_value=2)
    assert sweep_mod.main(["--base-url", "https://vb.org"]) == 1
    mocker.patch.object(sweep_mod, "sweep", return_value=0)
    assert sweep_mod.main(["--base-url", "https://vb.org"]) == 0
