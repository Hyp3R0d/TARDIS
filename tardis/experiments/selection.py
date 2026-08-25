"""Freeze deterministic benchmark record manifests without consulting test outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tardis.experiments.benchmark import DATASETS


def choose_record_ids(
    records: Sequence[Any],
    *,
    dataset: str,
    split: str,
    seed: int,
    count: int,
) -> list[str]:
    """Choose a catalog-order-independent subset using only public record identifiers."""

    if count <= 0:
        raise ValueError("count must be positive")
    record_ids = [str(record.id) for record in records]
    if any(not record_id for record_id in record_ids):
        raise ValueError("records contain an empty identifier")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("records contain duplicate identifiers")
    if count > len(record_ids):
        raise ValueError(f"requested {count} records from a split containing {len(record_ids)}")

    def rank(record_id: str) -> tuple[str, str]:
        payload = f"tardis-record-selection-v1\x1f{dataset}\x1f{split}\x1f{seed}\x1f{record_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), record_id

    return sorted(record_ids, key=rank)[:count]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--datasets-file", type=Path, default=Path("/home/TARDIS/datasets.txt"))
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    """Build a JSON manifest from a stable local dataset partition."""

    from tardis.cli.runtime import read_dataset_sources
    from tardis.data.assembly import RangeClientFactory, build_remote_catalog
    from tardis.data.catalog import normalize_local_dataset_roots
    from tardis.data.dataset import build_split_records
    from tardis.data.splits import StablePartition

    roots = normalize_local_dataset_roots(read_dataset_sources(args.datasets_file))
    client_factory = RangeClientFactory(
        max_object_bytes=128 * 1024 * 1024,
        timeout_seconds=float(args.request_timeout_seconds),
        max_retries=int(args.max_retries),
    )
    catalog = build_remote_catalog(
        client_factory=client_factory,
        dataset_roots=roots,
        selected_source=str(args.dataset),
    )
    splits = build_split_records(
        catalog.records_by_source,
        StablePartition(
            seed=int(args.split_seed),
            validation_size=int(args.validation_size),
            test_size=int(args.test_size),
            group_by_caption=str(args.dataset) == "seedance",
        ),
    )
    split = str(args.split)
    records = tuple(splits[split][str(args.dataset)])
    expected_size = int(args.validation_size if split == "validation" else args.test_size)
    if len(records) != expected_size:
        raise RuntimeError(
            f"locked {split} split has unexpected size: expected {expected_size}, "
            f"received {len(records)}"
        )
    record_ids = choose_record_ids(
        records,
        dataset=str(args.dataset),
        split=split,
        seed=int(args.split_seed),
        count=int(args.records),
    )
    digest = hashlib.sha256("\n".join(record_ids).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "split": split,
        "selection_algorithm": "sha256(tardis-record-selection-v1,dataset,split,seed,record_id)",
        "split_seed": int(args.split_seed),
        "validation_size": int(args.validation_size),
        "test_size": int(args.test_size),
        "source_split_size": len(records),
        "record_ids_sha256": digest,
        "records": [{"record_id": record_id} for record_id in record_ids],
    }


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if int(args.records) <= 0:
        raise ValueError("records must be positive")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"record manifest already exists: {output}")
    payload = build_manifest(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    main()
