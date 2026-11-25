import io
import os
import stat
import tarfile

import pytest

from tsgm.utils.file_utils import extract_archive


def _write_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for member, content in members:
            if member.isfile():
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            else:
                archive.addfile(member)


def test_extract_archive_accepts_nested_dataset_files(tmp_path):
    archive_path = tmp_path / "dataset.tar.gz"
    destination = tmp_path / "dataset"
    directory = tarfile.TarInfo("records")
    directory.type = tarfile.DIRTYPE
    _write_archive(archive_path, [
        (directory, b""),
        (tarfile.TarInfo("records/sample.csv"), b"time,value\n0,1\n"),
    ])

    extract_archive(str(archive_path), str(destination))

    assert (destination / "records/sample.csv").read_bytes() == b"time,value\n0,1\n"
    assert not archive_path.exists()


@pytest.mark.parametrize("escape", ["parent", "absolute", "sibling-prefix"])
def test_extract_archive_rejects_escaping_paths(tmp_path, escape):
    archive_path = tmp_path / "dataset.tar.gz"
    destination = tmp_path / "dataset"
    outside = tmp_path / "outside.txt"
    if escape == "sibling-prefix":
        # A string-prefix check would mistake this sibling for the destination.
        outside = tmp_path / "dataset-backup" / "outside.txt"
    member_name = str(outside) if escape == "absolute" else os.path.relpath(outside, destination)
    _write_archive(archive_path, [
        (tarfile.TarInfo("safe.csv"), b"data"),
        (tarfile.TarInfo(member_name), b"overwrite"),
    ])

    with pytest.raises(tarfile.ExtractError):
        extract_archive(str(archive_path), str(destination))

    assert not outside.exists()
    assert not (destination / "safe.csv").exists()
    assert archive_path.exists()


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE])
def test_extract_archive_rejects_links_and_special_files(tmp_path, member_type):
    archive_path = tmp_path / "dataset.tar.gz"
    destination = tmp_path / "dataset"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"original")
    member = tarfile.TarInfo("link")
    member.type = member_type
    member.linkname = "../outside.txt"
    _write_archive(archive_path, [
        (member, b""),
        (tarfile.TarInfo("link"), b"overwrite"),
    ])

    with pytest.raises(tarfile.ExtractError):
        extract_archive(str(archive_path), str(destination))

    assert outside.read_bytes() == b"original"
    assert archive_path.exists()


@pytest.mark.parametrize("link_kind", ["directory", "file"])
def test_extract_archive_rejects_existing_symlink_escape(tmp_path, link_kind):
    archive_path = tmp_path / "dataset.tar.gz"
    destination = tmp_path / "dataset"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "sample.csv"
    outside_file.write_bytes(b"original")
    if link_kind == "directory":
        (destination / "records").symlink_to(outside, target_is_directory=True)
    else:
        (destination / "records").mkdir()
        (destination / "records/sample.csv").symlink_to(outside_file)
    _write_archive(archive_path, [(tarfile.TarInfo("records/sample.csv"), b"overwrite")])

    with pytest.raises(tarfile.ExtractError):
        extract_archive(str(archive_path), str(destination))

    assert outside_file.read_bytes() == b"original"
    assert archive_path.exists()


def test_extract_archive_rejects_existing_hardlink_escape(tmp_path):
    archive_path = tmp_path / "dataset.tar.gz"
    destination = tmp_path / "dataset"
    destination.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"original")
    target = destination / "sample.csv"
    os.link(outside, target)
    _write_archive(archive_path, [(tarfile.TarInfo("sample.csv"), b"overwrite")])
    original_archive = archive_path.read_bytes()

    with pytest.raises(tarfile.ExtractError, match="hardlinked file"):
        extract_archive(str(archive_path), str(destination))

    assert outside.read_bytes() == b"original"
    assert target.read_bytes() == b"original"
    assert archive_path.read_bytes() == original_archive


def test_extract_archive_can_overwrite_existing_regular_file(tmp_path):
    archive_path = tmp_path / "dataset.tar.gz"
    destination = tmp_path / "dataset"
    destination.mkdir()
    target = destination / "sample.csv"
    target.write_bytes(b"original")
    _write_archive(archive_path, [(tarfile.TarInfo("sample.csv"), b"replacement")])

    extract_archive(str(archive_path), str(destination))

    assert target.read_bytes() == b"replacement"
    assert not archive_path.exists()


def test_extract_archive_retains_corrupt_archive(tmp_path):
    archive_path = tmp_path / "dataset.tar.gz"
    archive_path.write_bytes(b"not a gzip archive")

    with pytest.raises(tarfile.TarError):
        extract_archive(str(archive_path), str(tmp_path / "dataset"))

    assert archive_path.read_bytes() == b"not a gzip archive"


def test_extract_archive_retains_source_after_write_failure(tmp_path):
    archive_path = tmp_path / "dataset.tar.gz"
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "records").write_bytes(b"existing file")
    _write_archive(archive_path, [(tarfile.TarInfo("records/sample.csv"), b"data")])
    original = archive_path.read_bytes()

    with pytest.raises(OSError):
        extract_archive(str(archive_path), str(destination))

    assert archive_path.read_bytes() == original
    assert (destination / "records").read_bytes() == b"existing file"


def test_extract_archive_retains_source_after_metadata_failure(tmp_path, monkeypatch):
    archive_path = tmp_path / "dataset.tar.gz"
    _write_archive(archive_path, [(tarfile.TarInfo("sample.csv"), b"data")])
    original = archive_path.read_bytes()

    def fail_chmod(*args):
        raise tarfile.ExtractError("Cannot set file permissions")

    monkeypatch.setattr(tarfile.TarFile, "chmod", fail_chmod)

    with pytest.raises(tarfile.ExtractError, match="Cannot set file permissions"):
        extract_archive(str(archive_path), str(tmp_path / "dataset"))

    assert archive_path.read_bytes() == original


@pytest.mark.parametrize("source_alias", ["direct", "symlink", "hardlink"])
def test_extract_archive_rejects_overwriting_its_source(tmp_path, source_alias):
    archive_path = tmp_path / "dataset.tar.gz"
    target = archive_path if source_alias == "direct" else tmp_path / "archive-alias"
    _write_archive(archive_path, [(tarfile.TarInfo(target.name), b"overwrite")])
    if source_alias == "symlink":
        target.symlink_to(archive_path)
    elif source_alias == "hardlink":
        os.link(archive_path, target)
    original = archive_path.read_bytes()

    with pytest.raises(tarfile.ExtractError):
        extract_archive(str(archive_path))

    assert archive_path.read_bytes() == original
    assert target.read_bytes() == original


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_extract_archive_sanitizes_permissions_and_ownership(tmp_path):
    archive_path = tmp_path / "dataset.tar.gz"
    destination = tmp_path / "dataset"
    directory = tarfile.TarInfo("records")
    directory.type = tarfile.DIRTYPE
    directory.mode = 0o600
    member = tarfile.TarInfo("records/sample.csv")
    member.mode = 0o6777
    member.uid = member.gid = 12345
    _write_archive(archive_path, [(directory, b""), (member, b"data")])

    extract_archive(str(archive_path), str(destination))

    target = destination / "records/sample.csv"
    assert target.read_bytes() == b"data"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert target.stat().st_uid == os.getuid()
