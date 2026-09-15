"""Request guards for the optional web server, before multipart parsing."""

import secrets

from starlette.formparsers import MultiPartException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyTooLarge(MultiPartException):
    """Also triggers Starlette's cleanup of partially written upload files."""


class UploadGuard:
    """Authenticate writes and bound incoming bytes before parsing the body."""

    def __init__(self, app: ASGIApp, max_bytes: int, token: str = "") -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        if (self.token and request.url.path.startswith("/api/")
                and request.method in ("POST", "PUT", "PATCH", "DELETE")):
            authorization = request.headers.get("authorization", "")
            if not secrets.compare_digest(
                authorization.encode(), f"Bearer {self.token}".encode()
            ):
                await JSONResponse({"detail": "Invalid or missing token"}, 401)(scope, receive, send)
                return

        rejection = JSONResponse({"detail": "Upload too large"}, 413)
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await rejection(scope, receive, send)
            return

        size = 0
        exceeded = False
        rejected = False

        async def limited_receive() -> Message:
            nonlocal size, exceeded
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge("Upload too large")
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal rejected
            # FastAPI converts multipart parser exceptions to a 400. Replace
            # that response with 413 after the parser has closed its spools.
            if exceeded:
                if not rejected:
                    rejected = True
                    await rejection(scope, receive, send)
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            # Non-multipart readers may propagate the exception directly.
            if not rejected:
                await rejection(scope, receive, send)
