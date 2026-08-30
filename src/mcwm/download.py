"""Download a small official VPT demonstration pair."""

from __future__ import annotations

import urllib.request
from pathlib import Path

DEMO_STEM = "cheeky-cornflower-setter-02e496ce4abb-20220421-092639"
SECOND_DEMO_STEM = "cheeky-cornflower-setter-02e496ce4abb-20220421-093149"
DEMO_BASE_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0"


def download_url(
    url: str,
    destination: Path,
    force: bool = False,
    *,
    expected_bytes: int | None = None,
    show_progress: bool = True,
) -> Path:
    """Download atomically and optionally verify its manifest size."""
    if destination.exists() and not force:
        if expected_bytes is None or destination.stat().st_size == expected_bytes:
            print(f"Already present: {destination}")
            return destination
        raise ValueError(
            f"Existing file has the wrong size: {destination}; "
            "use --force to download it again"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "mcwm-learning-project/0.1"})

    print(f"Downloading {url}")
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        copied = 0
        next_report = 16 * 1024 * 1024
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            copied += len(chunk)
            if show_progress and total and (copied >= next_report or copied == total):
                print(f"  {copied / 1_048_576:7.1f}/{total / 1_048_576:.1f} MiB", end="\r")
                next_report += 16 * 1024 * 1024
    if show_progress and total:
        print()
    if expected_bytes is not None and copied != expected_bytes:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded size mismatch for {destination}: expected {expected_bytes}, got {copied}"
        )
    temporary.replace(destination)
    if not show_progress:
        print(f"Downloaded: {destination}")
    return destination


def download_episode(data_dir: Path, stem: str, force: bool = False) -> tuple[Path, Path]:
    """Download one matched VPT 10.0 video/action pair by episode stem."""
    if not stem or Path(stem).name != stem or Path(stem).suffix:
        raise ValueError("episode must be a plain filename stem")

    video_path = data_dir / f"{stem}.mp4"
    action_path = data_dir / f"{stem}.jsonl"
    download_url(f"{DEMO_BASE_URL}/{video_path.name}", video_path, force=force)
    download_url(f"{DEMO_BASE_URL}/{action_path.name}", action_path, force=force)
    return video_path, action_path


def download_demo(data_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """Download the exact video/action pair used by OpenAI's VPT demo."""
    return download_episode(data_dir, DEMO_STEM, force=force)
