"""Prepare the LatentGUIWorld SFT dataset for the training launcher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq

DATASET_REPO = "OpenMOSS-Team/LatentGUIWorld"


def extract_images(dataset_root: Path, archives):
    root = dataset_root.resolve()
    count = 0
    for archive in archives:
        archive_path = root / archive["path"]
        if archive_path.stat().st_size != archive["bytes"]:
            raise ValueError(f"Archive size differs from the dataset index: {archive['path']}")
        archive_count = 0
        with tarfile.open(archive_path, mode="r:") as handle:
            for member in handle:
                relative = PurePosixPath(member.name)
                if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Image archive contains an invalid entry")
                if relative.parts[:2] != ("training", "images"):
                    raise ValueError("Image archive entry is outside training/images")
                target = root.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(root):
                    raise ValueError("Image path is outside the dataset directory")
                if not target.is_file() or target.stat().st_size != member.size:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    stream = handle.extractfile(member)
                    if stream is None:
                        raise ValueError("Image archive entry has no data")
                    with stream, temporary.open("wb") as destination:
                        shutil.copyfileobj(stream, destination)
                    if temporary.stat().st_size != member.size:
                        raise ValueError("Image extraction was incomplete")
                    temporary.replace(target)
                archive_count += 1
        if archive_count != archive["images"]:
            raise ValueError("Image count differs from the dataset index")
        count += archive_count
    return count


def prepare(dataset_root: Path, output: Path):
    root = dataset_root.resolve()
    index = json.loads((root / "training/dataset.json").read_text())
    if output.exists():
        raise FileExistsError(output)
    image_count = extract_images(root, index["image_archives"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    writer = None
    count = 0
    try:
        for shard in index["data_shards"]:
            path = root / shard["path"]
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != shard["rows"]:
                raise ValueError("SFT shard size differs from the dataset index")
            for batch in parquet.iter_batches(batch_size=128):
                rows = batch.to_pylist()
                for row in rows:
                    if row["metadata"]["row_index"] != count:
                        raise ValueError("SFT row order differs from the dataset index")
                    for entry in row["images"]:
                        image = (root / entry["image"]).resolve()
                        if not image.is_relative_to(root / "training/images") or not image.is_file():
                            raise ValueError("SFT sample references an unavailable image")
                        entry["image"] = str(image)
                    count += 1
                table = pa.Table.from_pylist(rows, schema=parquet.schema_arrow)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
        if count != index["supervised_examples"]:
            raise ValueError("SFT example count differs from the dataset index")
        if writer is not None:
            writer.close()
            writer = None
        temporary.replace(output)
    finally:
        if writer is not None:
            writer.close()
    return {"examples": count, "images": image_count, "train_file": str(output.resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download the training files from Hugging Face")
    parser.add_argument("--revision", default="main", help="Hugging Face dataset revision")
    args = parser.parse_args(argv)
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", revision=args.revision,
                          local_dir=args.dataset_root, allow_patterns=["training/**", "LICENSE", "NOTICE.md"])
    print(json.dumps(prepare(args.dataset_root, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
