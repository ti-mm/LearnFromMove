"""Training download preparation preserves examples and resolves images."""
import io
import json
from pathlib import Path
import tarfile

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latentguiworld.training_data import extract_images, prepare


def fixture_data(root):
    image = io.BytesIO()
    Image.new("RGB", (16, 12), "blue").save(image, format="WEBP", lossless=True)
    name = "training/images/drag_egocentric/example/000.webp"
    archive = root / "training/archives/images.tar"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo(name)
        member.size = len(image.getvalue())
        handle.addfile(member, io.BytesIO(image.getvalue()))
    rows = [{"messages": [{"role": "user", "content": "<image> Move the shape."},
                           {"role": "assistant", "content": '<think>Move to the piece.</think>\n{"action":{"kind":"move_to","x":500,"y":500}}'}],
             "images": [{"image": name, "max_pixels": 921600}], "tools": [],
             "metadata": {"trajectory_id": "example", "variant": "drag_egocentric", "action_index": 0, "row_index": 0}}]
    shard = root / "training/data/train-00000.parquet"
    shard.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), shard)
    index = {"supervised_examples": 1,
             "data_shards": [{"path": str(shard.relative_to(root)), "rows": 1}],
             "image_archives": [{"path": str(archive.relative_to(root)), "images": 1, "bytes": archive.stat().st_size}]}
    (root / "training/dataset.json").write_text(json.dumps(index))
    return rows, index


def test_prepare_preserves_supervision_and_decodes_lossless_image(tmp_path):
    root = tmp_path / "dataset"
    rows, _ = fixture_data(root)
    output = tmp_path / "train.parquet"
    result = prepare(root, output)
    actual = pq.read_table(output).to_pylist()
    assert result["examples"] == result["images"] == 1
    assert actual[0]["messages"] == rows[0]["messages"]
    assert actual[0]["metadata"] == rows[0]["metadata"]
    path = Path(actual[0]["images"][0]["image"])
    assert path.is_absolute()
    assert Image.open(path).convert("RGB").getpixel((0, 0)) == (0, 0, 255)
    with pytest.raises(FileExistsError):
        prepare(root, output)


def test_wrong_order_is_rejected(tmp_path):
    rows, _ = fixture_data(tmp_path)
    rows[0]["metadata"]["row_index"] = 1
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "training/data/train-00000.parquet")
    with pytest.raises(ValueError, match="row order"):
        prepare(tmp_path, tmp_path / "train.parquet")


def test_archive_rejects_paths_outside_training_images(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        entry = tarfile.TarInfo("../escape.webp")
        entry.size = 1
        handle.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="invalid entry"):
        extract_images(tmp_path, [{"path": "bad.tar", "bytes": archive.stat().st_size, "images": 1}])
