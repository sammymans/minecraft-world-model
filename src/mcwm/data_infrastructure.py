"""Content-addressed dataset catalogs and safe S3-compatible publication."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from mcwm.dataset import ProcessedEpisode
from mcwm.manifest import DatasetManifest

CatalogSource = Literal["public_vpt", "local_recording"]


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash one file without loading a video or archive into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _relative_local_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"catalog input must be below source root {root}: {path}") from error


@dataclass(frozen=True)
class CatalogObject:
    """One immutable local file and its destination below a dataset prefix."""

    local_path: str
    object_key: str
    role: str
    bytes: int
    sha256: str
    media_type: str
    episode: str | None = None
    split: str | None = None
    source_url: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CatalogObject:
        return cls(**value)


@dataclass(frozen=True)
class DatasetCatalog:
    """A portable, content-verified publication unit for object storage."""

    schema_version: int
    catalog_id: str
    source_type: CatalogSource
    created_at: str
    source_root: str
    provenance: dict[str, Any]
    objects: tuple[CatalogObject, ...]

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported dataset catalog schema: {self.schema_version}")
        if not self.catalog_id or "/" in self.catalog_id:
            raise ValueError("catalog_id must be a non-empty object-key segment")
        if self.source_type not in {"public_vpt", "local_recording"}:
            raise ValueError(f"unsupported catalog source type: {self.source_type}")
        keys = [item.object_key for item in self.objects]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("catalog object keys must be present and unique")
        for item in self.objects:
            if item.bytes < 1:
                raise ValueError(f"catalog object is empty: {item.local_path}")
            if len(item.sha256) != 64:
                raise ValueError(f"catalog object has an invalid SHA-256: {item.local_path}")
            key_path = Path(item.object_key)
            if key_path.is_absolute() or ".." in key_path.parts:
                raise ValueError(f"unsafe catalog object key: {item.object_key}")

    @classmethod
    def load(cls, path: Path) -> DatasetCatalog:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read dataset catalog: {path}") from error
        value["objects"] = tuple(CatalogObject.from_dict(item) for item in value["objects"])
        catalog = cls(**value)
        catalog.validate()
        return catalog

    def write(self, path: Path) -> Path:
        self.validate()
        value = asdict(self)
        value["objects"] = [asdict(item) for item in self.objects]
        _atomic_json(path, value)
        return path

    def resolve(self, item: CatalogObject, *, source_root: Path | None = None) -> Path:
        root = source_root.resolve() if source_root else Path(self.source_root)
        path = (root / item.local_path).resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"catalog path escapes source root: {item.local_path}")
        return path


def _catalog_object(
    path: Path,
    root: Path,
    object_key: str,
    role: str,
    *,
    episode: str | None = None,
    split: str | None = None,
    source_url: str | None = None,
) -> CatalogObject:
    if not path.is_file():
        raise ValueError(f"catalog input is missing: {path}")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if path.suffix == ".jsonl":
        media_type = "application/x-ndjson"
    elif path.suffix == ".npz":
        media_type = "application/x-npz"
    return CatalogObject(
        local_path=_relative_local_path(path, root),
        object_key=object_key,
        role=role,
        bytes=path.stat().st_size,
        sha256=file_sha256(path),
        media_type=media_type,
        episode=episode,
        split=split,
        source_url=source_url,
    )


def build_public_dataset_catalog(
    manifest_path: Path,
    raw_dir: Path,
    processed_dir: Path,
    output_path: Path,
    *,
    source_root: Path,
    split: str = "all",
    include_raw: bool = True,
    include_processed: bool = True,
    target_fps: float = 10.0,
    image_size: int = 64,
    horizon: int = 8,
) -> DatasetCatalog:
    """Catalog a complete public-data snapshot before it is published."""
    if not include_raw and not include_processed:
        raise ValueError("catalog must include raw files, processed files, or both")
    manifest = DatasetManifest.load(manifest_path)
    selected = manifest.select(split)
    if not selected:
        raise ValueError(f"manifest has no episodes for split {split}")
    manifest_hash = file_sha256(manifest_path)
    split_label = "all" if split == "all" else split
    catalog_id = (
        f"{manifest_path.stem}-{split_label}-{manifest_hash[:12]}-"
        f"{target_fps:g}hz-{image_size}px-h{horizon}"
    )
    objects = [
        _catalog_object(
            manifest_path,
            source_root,
            f"manifests/{manifest_path.name}",
            "dataset_manifest",
        )
    ]
    for index, entry in enumerate(selected, 1):
        print(f"Cataloging [{index}/{len(selected)}]: {entry.episode}")
        if include_raw:
            video_path, action_path = entry.raw_paths(raw_dir)
            if video_path.is_file() and video_path.stat().st_size != entry.video_bytes:
                raise ValueError(f"raw video size does not match manifest: {video_path}")
            if action_path.is_file() and action_path.stat().st_size != entry.actions_bytes:
                raise ValueError(f"raw action size does not match manifest: {action_path}")
            objects.extend(
                (
                    _catalog_object(
                        video_path,
                        source_root,
                        f"raw/{entry.split}/{video_path.name}",
                        "raw_video",
                        episode=entry.episode,
                        split=entry.split,
                        source_url=entry.video_url,
                    ),
                    _catalog_object(
                        action_path,
                        source_root,
                        f"raw/{entry.split}/{action_path.name}",
                        "raw_actions",
                        episode=entry.episode,
                        split=entry.split,
                        source_url=entry.actions_url,
                    ),
                )
            )
        if include_processed:
            processed_path = entry.processed_path(processed_dir)
            processed = ProcessedEpisode.load(processed_path)
            if processed.episode != entry.episode:
                raise ValueError(f"processed episode identity mismatch: {processed_path}")
            objects.append(
                _catalog_object(
                    processed_path,
                    source_root,
                    f"processed/{entry.split}/{processed_path.name}",
                    "processed_episode",
                    episode=entry.episode,
                    split=entry.split,
                )
            )
    catalog = DatasetCatalog(
        schema_version=1,
        catalog_id=catalog_id,
        source_type="public_vpt",
        created_at=datetime.now(UTC).isoformat(),
        source_root=str(source_root.resolve()),
        provenance={
            "manifest": _relative_local_path(manifest_path, source_root),
            "manifest_sha256": manifest_hash,
            "episode_count": len(selected),
            "split": split,
            "preprocessing": {
                "target_fps": target_fps,
                "image_size": image_size,
                "horizon": horizon,
            },
        },
        objects=tuple(objects),
    )
    catalog.write(output_path)
    return catalog


def build_recording_catalog(
    metadata_path: Path,
    output_path: Path,
    *,
    source_root: Path,
    processed_path: Path | None = None,
) -> DatasetCatalog:
    """Catalog one honestly recorded local episode and its derived archive."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        episode = str(metadata["episode"])
        video_path = source_root / metadata["video_path"]
        actions_path = source_root / metadata["actions_path"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"invalid recording metadata: {metadata_path}") from error
    prefix = f"recordings/{episode}"
    objects = [
        _catalog_object(
            metadata_path, source_root, f"{prefix}/recording.json", "recording_metadata"
        ),
        _catalog_object(
            video_path,
            source_root,
            f"{prefix}/raw/{video_path.name}",
            "raw_video",
            episode=episode,
        ),
        _catalog_object(
            actions_path,
            source_root,
            f"{prefix}/raw/{actions_path.name}",
            "raw_actions",
            episode=episode,
        ),
    ]
    if processed_path is not None:
        processed = ProcessedEpisode.load(processed_path)
        if processed.episode != episode:
            raise ValueError("processed episode does not match recording metadata")
        objects.append(
            _catalog_object(
                processed_path,
                source_root,
                f"{prefix}/processed/{processed_path.name}",
                "processed_episode",
                episode=episode,
            )
        )
    catalog = DatasetCatalog(
        schema_version=1,
        catalog_id=f"local-{episode}",
        source_type="local_recording",
        created_at=datetime.now(UTC).isoformat(),
        source_root=str(source_root.resolve()),
        provenance={
            "recording_metadata": _relative_local_path(metadata_path, source_root),
            "collector": metadata.get("collector"),
            "capture_started_at": metadata.get("started_at"),
            "episode": episode,
        },
        objects=tuple(objects),
    )
    catalog.write(output_path)
    return catalog


class S3Client(Protocol):
    def head_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def upload_file(self, *args: Any, **kwargs: Any) -> None: ...


@dataclass(frozen=True)
class PublishResult:
    destination: str
    planned: int
    uploaded: int
    skipped: int
    bytes_uploaded: int
    dry_run: bool


def _join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def create_s3_client(*, endpoint_url: str | None = None, region: str | None = None) -> S3Client:
    """Create the client lazily so catalog-only workflows need no AWS setup."""
    import boto3

    return boto3.client("s3", endpoint_url=endpoint_url, region_name=region)


def _object_is_current(client: S3Client, bucket: str, key: str, item: CatalogObject) -> bool:
    try:
        remote = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # botocore client errors are optional at import time
        response = getattr(error, "response", {})
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = str(response.get("Error", {}).get("Code", ""))
        if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    metadata = {str(key).lower(): str(value) for key, value in remote.get("Metadata", {}).items()}
    return (
        int(remote.get("ContentLength", -1)) == item.bytes and metadata.get("sha256") == item.sha256
    )


def publish_catalog(
    catalog_path: Path,
    bucket: str,
    prefix: str,
    *,
    source_root: Path | None = None,
    execute: bool = False,
    workers: int = 4,
    endpoint_url: str | None = None,
    region: str | None = None,
    storage_class: str | None = None,
    sse: str | None = None,
    kms_key_id: str | None = None,
    client: S3Client | None = None,
) -> PublishResult:
    """Upload immutable objects and write the catalog last as a commit marker.

    Publication is a dry run unless ``execute`` is true. Existing objects are
    skipped only when both byte size and the stored SHA-256 metadata match. The
    command never deletes remote objects.
    """
    if not bucket or "/" in bucket:
        raise ValueError("bucket must be an S3 bucket name, not a URI")
    if workers < 1:
        raise ValueError("upload workers must be positive")
    if kms_key_id and sse != "aws:kms":
        raise ValueError("kms_key_id requires sse='aws:kms'")
    catalog = DatasetCatalog.load(catalog_path)
    root_key = _join_key(prefix, "datasets", catalog.catalog_id)
    destination = f"s3://{bucket}/{root_key}/"
    if not execute:
        return PublishResult(destination, len(catalog.objects) + 1, 0, 0, 0, True)
    if client is None:
        client = create_s3_client(endpoint_url=endpoint_url, region=region)

    def upload(item: CatalogObject) -> tuple[bool, int]:
        key = _join_key(root_key, item.object_key)
        if _object_is_current(client, bucket, key, item):
            return False, 0
        extra: dict[str, Any] = {
            "ContentType": item.media_type,
            "Metadata": {
                "sha256": item.sha256,
                "role": item.role,
                "catalog-id": catalog.catalog_id,
            },
        }
        if storage_class:
            extra["StorageClass"] = storage_class
        if sse:
            extra["ServerSideEncryption"] = sse
        if kms_key_id:
            extra["SSEKMSKeyId"] = kms_key_id
        local_path = catalog.resolve(item, source_root=source_root)
        if not local_path.is_file() or local_path.stat().st_size != item.bytes:
            raise ValueError(f"cataloged local object changed or disappeared: {local_path}")
        if file_sha256(local_path) != item.sha256:
            raise ValueError(f"cataloged local object checksum changed: {local_path}")
        client.upload_file(str(local_path), bucket, key, ExtraArgs=extra)
        return True, item.bytes

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(upload, catalog.objects))
    uploaded = sum(int(changed) for changed, _ in outcomes)
    uploaded_bytes = sum(size for _, size in outcomes)
    skipped = len(outcomes) - uploaded

    catalog_hash = file_sha256(catalog_path)
    catalog_key = _join_key(root_key, "_catalog.json")
    catalog_item = CatalogObject(
        local_path=str(catalog_path),
        object_key="_catalog.json",
        role="catalog_commit",
        bytes=catalog_path.stat().st_size,
        sha256=catalog_hash,
        media_type="application/json",
    )
    if not _object_is_current(client, bucket, catalog_key, catalog_item):
        extra = {
            "ContentType": "application/json",
            "Metadata": {"sha256": catalog_hash, "role": "catalog_commit"},
        }
        if storage_class:
            extra["StorageClass"] = storage_class
        if sse:
            extra["ServerSideEncryption"] = sse
        if kms_key_id:
            extra["SSEKMSKeyId"] = kms_key_id
        client.upload_file(str(catalog_path), bucket, catalog_key, ExtraArgs=extra)
        uploaded += 1
        uploaded_bytes += catalog_path.stat().st_size
    else:
        skipped += 1
    return PublishResult(
        destination=destination,
        planned=len(catalog.objects) + 1,
        uploaded=uploaded,
        skipped=skipped,
        bytes_uploaded=uploaded_bytes,
        dry_run=False,
    )
