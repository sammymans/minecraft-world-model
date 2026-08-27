from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch import nn

from mcwm.frontend import WebRolloutController, _handler
from mcwm.interactive import InteractiveRolloutEngine, RolloutSeed


class _MouseDynamics(nn.Module):
    latent_dim = 1
    action_dim = 9

    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        del previous_latent
        return current_latent + action[:, 7:8].div(100)


class _GrayDecoder(nn.Module):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents[:, :, None, None].expand(-1, 3, 2, 2)


def _controller() -> WebRolloutController:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    engine = InteractiveRolloutEngine(
        _GrayDecoder(),
        _MouseDynamics(),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        frame,
        torch.device("cpu"),
    )
    seed = RolloutSeed("web-test", 12, 10, frame, frame)
    return WebRolloutController(
        engine,
        seed,
        seed_index=3,
        seed_count=9,
        camera_step=30,
    )


def test_web_controller_steps_and_resets_the_recursive_engine() -> None:
    controller = _controller()

    _, step, action = controller.step({"controls": ["look_right"]})

    assert step == 1
    assert "mouse +30" in action
    _, reset_step = controller.reset()
    assert reset_step == 0


def test_local_frontend_serves_single_view_and_model_endpoints() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(_controller()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base, timeout=2) as response:  # noqa: S310
            html = response.read().decode()
        assert html.count('id="frame"') == 1
        assert "real seed t-1" not in html
        assert "real seed t" not in html

        request = Request(  # noqa: S310
            f"{base}/api/step",
            data=json.dumps({"controls": ["w"], "mouse_dx": 4}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:  # noqa: S310
            assert response.headers["X-Rollout-Step"] == "1"
            assert response.read().startswith(b"\x89PNG")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
