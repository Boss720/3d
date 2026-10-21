"""Process-wide admission control for requests that receive uploaded bodies."""

from __future__ import annotations

import json
import threading
from collections.abc import Set


DEFAULT_UPLOAD_PATHS = frozenset({
    "/api/jobs",
    "/api/model",
    "/api/preview",
    "/api/calibration",
})


class UploadAdmissionMiddleware:
    """Bound concurrently admitted upload requests across threads and ASGI loops.

    This middleware deliberately does not consume or inspect request bodies. The
    existing request-size middleware remains responsible for byte limits.
    """

    def __init__(self, app, paths: Set[str] | None = None, max_inflight: int = 4):
        if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight <= 0:
            raise ValueError("max_inflight must be a positive integer")
        if paths is None:
            recognized = DEFAULT_UPLOAD_PATHS
        else:
            if not isinstance(paths, Set) or isinstance(paths, (str, bytes)):
                raise TypeError("paths must be a set of absolute path strings")
            if any(not isinstance(path, str) or not path.startswith("/") or "?" in path or "#" in path for path in paths):
                raise ValueError("paths must contain absolute URL paths without query or fragment")
            recognized = frozenset(paths)

        self.app = app
        self.paths = recognized
        self.max_inflight = max_inflight
        self._lock = threading.Lock()
        self._inflight = 0

    def _try_acquire(self) -> bool:
        with self._lock:
            if self._inflight >= self.max_inflight:
                return False
            self._inflight += 1
            return True

    def _release(self) -> None:
        with self._lock:
            self._inflight -= 1

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http"
                or scope.get("method") != "POST"
                or scope.get("path") not in self.paths):
            await self.app(scope, receive, send)
            return

        if not self._try_acquire():
            body = json.dumps({
                "detail": "Too many uploads are being received. Please retry shortly."
            }, separators=(",", ":")).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            })
            await send({"type": "http.response.body", "body": body, "more_body": False})
            return

        try:
            await self.app(scope, receive, send)
        finally:
            self._release()
