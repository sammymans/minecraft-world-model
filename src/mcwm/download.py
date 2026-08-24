"""Download action/state JSONL files from an official VPT index."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

DEFAULT_INDEX_URL = (
    "https://openaipublic.blob.core.windows.net/minecraft-rl/"
    "snapshots/all_10xx_Jun_29.json"
)


def _load_index(source: str) -> dict[str, object]:
    path = Path(source)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    with urllib.request.urlopen(source) as response:  # noqa: S310
        return json.load(response)


def download_vpt_actions(
    output_directory: str | Path,
    *,
    index_source: str = DEFAULT_INDEX_URL,
    limit: int = 6,
    start: int = 0,
) -> list[Path]:
    index = _load_index(index_source)
    base_url = str(index["basedir"])
    relpaths = [
        str(path) for path in index["relpaths"] if str(path).endswith(".mp4")
    ]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    candidates = relpaths[start:]
    for number, relpath in enumerate(candidates, start=1):
        if len(downloaded) >= limit:
            break
        jsonl_relpath = relpath.removesuffix(".mp4") + ".jsonl"
        destination = output_directory / Path(jsonl_relpath).name
        if destination.exists():
            print(f"[{len(downloaded) + 1}/{limit}] exists: {destination.name}")
            downloaded.append(destination)
            continue
        url = urljoin(base_url, jsonl_relpath)
        temporary = destination.with_suffix(".jsonl.part")
        print(f"[{len(downloaded) + 1}/{limit}] downloading: {destination.name}")
        try:
            urllib.request.urlretrieve(url, temporary)  # noqa: S310
        except urllib.error.HTTPError as exc:
            temporary.unlink(missing_ok=True)
            if exc.code == 404:
                print(f"  unavailable in public blob; skipping index item {number}")
                continue
            raise
        temporary.replace(destination)
        downloaded.append(destination)
    if len(downloaded) < limit:
        print(f"warning: only {len(downloaded)} of {limit} requested recordings exist")
    return downloaded
