import json
import urllib.error
from pathlib import Path

from mcwm.download import download_vpt_actions


def test_downloader_skips_missing_public_blobs(
    tmp_path: Path, monkeypatch
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "basedir": "https://example.test/data/",
                "relpaths": ["missing.mp4", "first.mp4", "second.mp4"],
            }
        ),
        encoding="utf-8",
    )

    def fake_retrieve(url: str, destination: str | Path) -> None:
        if url.endswith("missing.jsonl"):
            raise urllib.error.HTTPError(url, 404, "missing", {}, None)
        Path(destination).write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_retrieve)
    downloaded = download_vpt_actions(
        tmp_path / "episodes", index_source=str(index_path), limit=2
    )
    assert [path.name for path in downloaded] == ["first.jsonl", "second.jsonl"]
