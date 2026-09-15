"""Regression tests for request guards before FastAPI parses upload bodies."""

import asyncio
import copy
from pathlib import Path
from tempfile import SpooledTemporaryFile

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from starlette import formparsers  # noqa: E402

from slide_stream import serve  # noqa: E402
from slide_stream.config_loader import DEFAULT_CONFIG  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_server(monkeypatch):
    for name in ("SLIDESTREAM_LOCAL", "SLIDESTREAM_DEMO", "SLIDESTREAM_TOKEN",
                 "SLIDESTREAM_TRUSTED_PROXY", "SLIDESTREAM_TRUSTED_PROXIES"):
        monkeypatch.delenv(name, raising=False)
    serve._DEMO_HITS.clear()
    serve._JOBS.clear()
    serve._PROJECTS.clear()
    monkeypatch.setattr(serve.ThreadPoolExecutor, "submit", lambda *a, **k: None)
    yield
    serve._DEMO_HITS.clear()
    serve._JOBS.clear()
    serve._PROJECTS.clear()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_local_settings_reject_rebound_host(method):
    (Path.home() / ".slidestream.yaml").write_text("secret: sentinel")
    client = TestClient(serve.create_app(config=copy.deepcopy(DEFAULT_CONFIG), local=True),
                        base_url="http://attacker.example:8080")
    response = client.request(method, "/api/settings")
    assert response.status_code == 403
    assert "sentinel" not in response.text


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]", "[::1]:8080"])
def test_local_settings_accept_loopback_hosts(host):
    client = TestClient(serve.create_app(config=copy.deepcopy(DEFAULT_CONFIG), local=True))
    assert client.get("/api/settings", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("endpoint", ["/api/jobs", "/api/check", "project"])
def test_avatar_name_cannot_be_a_server_path(endpoint, tmp_path):
    image = tmp_path / "private.png"
    image.write_bytes(b"private image")
    demo = endpoint != "project"
    client = TestClient(serve.create_app(config=copy.deepcopy(DEFAULT_CONFIG), demo=demo))
    files = {"deck": ("deck.md", b"# Slide\n- Bullet")}
    headers = {}
    if endpoint == "project":
        project = client.post("/api/projects", files=files).json()
        endpoint = f"/api/projects/{project['project_id']}/render"
        headers = {"X-Project-Token": project["token"]}
        files = {}
    response = client.post(endpoint, files=files, headers=headers,
                           data={"avatar_name": str(image), "avatar": "false"})
    assert response.status_code == 400
    assert not serve._JOBS


def _asgi_request(app, headers, chunks):
    """Send distinct ASGI chunks without a client buffering them together."""
    messages = []
    consumed = []

    async def run():
        iterator = iter(chunks)

        async def receive():
            body, more = next(iterator)
            consumed.append(body)
            return {"type": "http.request", "body": body, "more_body": more}

        async def send(message):
            messages.append(message)

        await app({"type": "http", "asgi": {"version": "3.0"},
                   "http_version": "1.1", "method": "POST", "scheme": "http",
                   "path": "/api/jobs", "raw_path": b"/api/jobs",
                   "query_string": b"", "root_path": "", "headers": [(b"host", b"localhost"), *headers],
                   "client": ("192.0.2.10", 1234), "server": ("localhost", 8080)},
                  receive, send)

    asyncio.run(run())
    return messages, consumed


def test_unauthorized_upload_rejected_without_receiving_body():
    app = serve.create_app(config=copy.deepcopy(DEFAULT_CONFIG), token="secret")
    messages, consumed = _asgi_request(app, [(b"content-type", b"multipart/form-data; boundary=x")], [])
    assert messages[0]["status"] == 401
    assert consumed == []


def test_declared_oversize_rejected_without_receiving_body():
    app = serve.create_app(config=copy.deepcopy(DEFAULT_CONFIG))
    messages, consumed = _asgi_request(app, [(b"content-length", b"999999999999")], [])
    assert messages[0]["status"] == 413
    assert consumed == []


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("declared", [[], [(b"content-length", b"1")]])
def test_chunked_upload_limit_closes_partial_spools(monkeypatch, local, declared):
    monkeypatch.setattr(serve, "MAX_DECK_BYTES", 128)
    monkeypatch.setattr(serve, "MAX_VOICE_BYTES", 128)
    monkeypatch.setattr(serve, "MAX_PHOTO_BYTES", 128)
    spools = []

    def spool(*args, **kwargs):
        result = SpooledTemporaryFile(*args, **kwargs)
        spools.append(result)
        return result

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", spool)
    app = serve.create_app(config=copy.deepcopy(DEFAULT_CONFIG), local=local)
    first = b'--x\r\nContent-Disposition: form-data; name="deck"; filename="deck.md"\r\n\r\n# Slide\n'
    messages, consumed = _asgi_request(
        app, [(b"content-type", b"multipart/form-data; boundary=x"), *declared],
        [(first, True), (b"x" * (1024 * 1024 + 1024), True), (b"\r\n--x--\r\n", False)],
    )
    assert messages[0]["status"] == 413
    assert len(consumed) == 2
    assert spools and all(s.closed for s in spools)
    assert not serve._JOBS


@pytest.mark.parametrize("proxies", ["", "10.0.0.0/24"])
def test_untrusted_peer_cannot_rotate_forwarded_ip(monkeypatch, proxies):
    monkeypatch.setenv("SLIDESTREAM_TRUSTED_PROXY", "1")  # old switch must not trust everyone
    monkeypatch.setenv("SLIDESTREAM_TRUSTED_PROXIES", proxies)
    client = TestClient(serve.create_app(config=copy.deepcopy(DEFAULT_CONFIG), demo=True),
                        client=("192.0.2.10", 1234))
    for i in range(serve.DEMO_JOBS_PER_HOUR + 1):
        response = client.post("/api/jobs", files={"deck": ("deck.md", b"# Slide")},
                               headers={"X-Forwarded-For": f"198.51.100.{i}"})
        assert response.status_code == (200 if i < serve.DEMO_JOBS_PER_HOUR else 429)
