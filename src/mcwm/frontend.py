"""Dependency-free local web frontend for the interactive rollout engine."""

from __future__ import annotations

import json
import math
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from mcwm.interactive import (
    BINARY_ACTIONS,
    InteractiveRolloutEngine,
    PlaygroundResult,
    RolloutSeed,
    make_live_action,
)

FRONTEND_PATH = Path(__file__).parent / "web" / "rollout.html"
LOOK_CONTROLS = {"look_left", "look_right", "look_up", "look_down"}
ALLOWED_CONTROLS = set(BINARY_ACTIONS) | LOOK_CONTROLS


@dataclass(frozen=True)
class FrontendInfo:
    episode: str
    seed_step: int
    seed_index: int
    seed_count: int
    model_fps: float
    device: str


class WebRolloutController:
    """Thread-safe model state exposed through the local HTTP handler."""

    def __init__(
        self,
        engine: InteractiveRolloutEngine,
        seed: RolloutSeed,
        *,
        seed_index: int,
        seed_count: int,
        camera_step: float,
    ) -> None:
        if camera_step <= 0:
            raise ValueError("camera_step must be positive")
        self.engine = engine
        self.seed = seed
        self.seed_index = seed_index
        self.seed_count = seed_count
        self.camera_step = camera_step
        self.lock = threading.Lock()

    @property
    def info(self) -> FrontendInfo:
        return FrontendInfo(
            episode=self.seed.episode,
            seed_step=self.seed.current_step,
            seed_index=self.seed_index,
            seed_count=self.seed_count,
            model_fps=self.seed.model_fps,
            device=str(self.engine.device),
        )

    @staticmethod
    def _png(frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("could not encode rollout frame")
        return encoded.tobytes()

    def current_png(self) -> tuple[bytes, int]:
        with self.lock:
            return self._png(self.engine.current_frame), self.engine.steps

    def reset(self) -> tuple[bytes, int]:
        with self.lock:
            frame = self.engine.reset()
            return self._png(frame), self.engine.steps

    def step(self, payload: dict[str, Any]) -> tuple[bytes, int, str]:
        raw_controls = payload.get("controls", [])
        if not isinstance(raw_controls, list) or not all(
            isinstance(item, str) for item in raw_controls
        ):
            raise ValueError("controls must be a list of strings")
        controls = {item.lower() for item in raw_controls}
        unknown = controls - ALLOWED_CONTROLS
        if unknown:
            raise ValueError(f"unknown controls: {', '.join(sorted(unknown))}")
        mouse_dx = _finite_mouse_delta(payload.get("mouse_dx", 0))
        mouse_dy = _finite_mouse_delta(payload.get("mouse_dy", 0))
        action = make_live_action(
            controls,
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            camera_step=self.camera_step,
        )
        with self.lock:
            frame = self.engine.step(action)
            return self._png(frame), self.engine.steps, _action_text(action)


def _finite_mouse_delta(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("mouse deltas must be numbers") from error
    if not math.isfinite(parsed):
        raise ValueError("mouse deltas must be finite")
    return float(np.clip(parsed, -500, 500))


def _action_text(action: np.ndarray) -> str:
    keys = [
        name.upper()
        for name, value in zip(BINARY_ACTIONS, action[:7], strict=True)
        if value
    ]
    key_text = "+".join(keys) if keys else "IDLE"
    return f"{key_text} · mouse {action[-2]:+.0f}, {action[-1]:+.0f}"


def _handler(controller: WebRolloutController) -> type[BaseHTTPRequestHandler]:
    html = FRONTEND_PATH.read_bytes()

    class RolloutHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _send(
            self,
            body: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            *,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(
                (json.dumps(value) + "\n").encode(),
                "application/json; charset=utf-8",
                status,
            )

        def _frame(self, frame: bytes, step: int, action: str | None = None) -> None:
            headers = {"X-Rollout-Step": str(step)}
            if action is not None:
                headers["X-Rollout-Action"] = action
            self._send(frame, "image/png", headers=headers)

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/":
                self._send(html, "text/html; charset=utf-8")
            elif path == "/api/info":
                self._json(controller.info.__dict__)
            elif path == "/api/frame":
                self._frame(*controller.current_png())
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                if path == "/api/step":
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > 16_384:
                        raise ValueError("request body is too large")
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("request body must be a JSON object")
                    self._frame(*controller.step(payload))
                elif path == "/api/reset":
                    self._frame(*controller.reset())
                elif path == "/api/stop":
                    self._json({"stopping": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return RolloutHandler


def serve_rollout_frontend(
    engine: InteractiveRolloutEngine,
    seed: RolloutSeed,
    *,
    seed_index: int,
    seed_count: int,
    camera_step: float,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> PlaygroundResult:
    """Serve one local browser session until Ctrl-C or the Stop button."""
    controller = WebRolloutController(
        engine,
        seed,
        seed_index=seed_index,
        seed_count=seed_count,
        camera_step=camera_step,
    )
    server = ThreadingHTTPServer((host, port), _handler(controller))
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}"
    print(f"frontend:       {url}")
    print("stop:           Ctrl-C in this terminal, or click Stop server")
    if open_browser:
        threading.Timer(0.2, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return PlaygroundResult(
        episode=seed.episode,
        current_step=seed.current_step,
        steps=engine.steps,
        device=str(engine.device),
    )
