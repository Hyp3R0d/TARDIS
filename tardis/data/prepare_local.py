"""Build compact revision-pinned manifests for the downloaded local subsets."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from tardis.data.catalog import (
    LOCAL_DATASET_DIRECTORIES,
    LOCAL_MANIFEST_NAME,
    build_local_adapters,
    normalize_local_dataset_roots,
)
from tardis.data.contracts import VideoRecord
from tardis.data.http_range import RangeClient

_DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024


def prepare_local_manifests(
    dataset_roots: Mapping[str, Path | str],
    *,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> dict[str, int]:
    """Filter raw metadata once and atomically publish all three manifests."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    roots = {
        source: Path(dataset_roots[source]).expanduser().resolve()
        for source in LOCAL_DATASET_DIRECTORIES
    }
    temporary = {
        source: root / f".{LOCAL_MANIFEST_NAME}.tmp-{os.getpid()}" for source, root in roots.items()
    }
    counts: dict[str, int] = {}
    try:
        with RangeClient(max_object_bytes=max(128 * 1024 * 1024, chunk_bytes)) as client:
            adapters = build_local_adapters(
                client,
                dataset_roots=roots,
                chunk_bytes=chunk_bytes,
                prefer_manifests=False,
            )
            for source in LOCAL_DATASET_DIRECTORIES:
                seen: set[str] = set()
                count = 0
                with temporary[source].open("w", encoding="utf-8", newline="\n") as stream:
                    for record in adapters[source].iter_records():
                        if record.id in seen:
                            raise ValueError(
                                "local "
                                f"{source} metadata contains duplicate record ID {record.id!r}"
                            )
                        seen.add(record.id)
                        stream.write(
                            json.dumps(
                                _record_payload(record),
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                        )
                        stream.write("\n")
                        count += 1
                    stream.flush()
                    os.fsync(stream.fileno())
                if count == 0:
                    raise ValueError(f"local {source} manifest would be empty")
                counts[source] = count

        for source in LOCAL_DATASET_DIRECTORIES:
            os.replace(temporary[source], roots[source] / LOCAL_MANIFEST_NAME)
        return counts
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def _record_payload(record: VideoRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "caption": record.caption,
        "media_locator": record.media_locator,
        "source": record.source,
        "metadata": record.metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TARDIS local dataset manifests")
    parser.add_argument("--datasets-file", type=Path, default=Path("datasets.txt"))
    parser.add_argument("--chunk-bytes", type=int, default=_DEFAULT_CHUNK_BYTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.datasets_file.is_file():
        raise FileNotFoundError(f"dataset source file does not exist: {args.datasets_file}")
    sources = tuple(
        line.strip()
        for line in args.datasets_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    roots = normalize_local_dataset_roots(sources)
    counts = prepare_local_manifests(roots, chunk_bytes=int(args.chunk_bytes))
    for source in LOCAL_DATASET_DIRECTORIES:
        print(f"{source}: {counts[source]} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
