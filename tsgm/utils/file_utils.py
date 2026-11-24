import os
import copy
import zipfile
import typing as T
import hashlib
import logging
import tarfile
import urllib

import urllib.request


logger = logging.getLogger('utils')
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def _archive_type(file: str) -> str:
    _, ext = os.path.splitext(file)
    if ext in EXTRACTORS.keys():
        return ext
    else:
        raise ValueError(f"Unsupported Extension: {ext}")


def _extract_zip(from_path: str, to_path: str, pwd: T.Optional[bytes]) -> None:
    with zipfile.ZipFile(from_path, "r", compression=zipfile.ZIP_STORED) as zip:
        zip.extractall(to_path, pwd=pwd)


def _extract_targz(from_path: str, to_path: str, pwd: T.Optional[bytes] = None) -> None:
    """Extract dataset files/directories, rejecting links and special files."""
    destination = os.path.realpath(to_path)
    source = os.path.realpath(from_path)
    with tarfile.open(from_path, "r:gz") as tar:
        members = [copy.copy(member) for member in tar.getmembers()]
        for member in members:
            # Dataset archives need only files and directories. Reject links as
            # well as devices/FIFOs, including on Python without data_filter.
            if not (member.isfile() or member.isdir()):
                raise tarfile.ExtractError(f"Unsupported archive member: {member.name!r}")
            target = os.path.realpath(os.path.join(destination, member.name))
            if os.path.isabs(member.name) or os.path.commonpath((destination, target)) != destination:
                raise tarfile.ExtractError(f"Archive member escapes destination: {member.name!r}")
            if target == source or (os.path.exists(target) and os.path.samefile(target, source)):
                raise tarfile.ExtractError(f"Archive member would overwrite its source: {member.name!r}")
            # Use numeric values supported by older Python tarfile versions.
            # Never restore archive ownership or setuid/setgid/sticky bits.
            member.uid = member.gid = -1
            member.uname = member.gname = ""
            member.mode = (member.mode & 0o755) | (0o700 if member.isdir() else 0o600)

        # Do not suppress extraction failures: extract_archive must retain the
        # original archive unless extraction has completed successfully.
        tar.errorlevel = 2
        tar.extractall(to_path, members=members)
    logger.info("Files extracted successfully.")


EXTRACTORS = {
    #  TODO add ".tar" & ".bz2"
    ".zip": _extract_zip,
    ".gz": _extract_targz,
}


def extract_archive(from_path: str, to_path: T.Optional[str] = None, pwd: T.Optional[bytes] = None) -> None:
    ext = _archive_type(from_path)
    extractor = EXTRACTORS[ext]

    if to_path is None:
        to_path, _ = os.path.splitext(from_path)
        to_path = os.path.dirname(to_path)

    extractor(from_path, to_path, pwd=pwd)
    os.remove(from_path)


def download(url: str, path: str, md5: T.Optional[str] = None, max_attempt: int = 3) -> None:
    logger.info(f"### Downloading from {url} ###")
    os.makedirs(path, exist_ok=True)
    if "?" in url:
        base_url, _ = url.split("?")
    else:
        base_url = url
    resource_name = base_url.split("/")[-1]
    path = os.path.join(path, resource_name)
    for attempt in range(max_attempt):
        logger.info(f"Attempt {attempt + 1} / {max_attempt}")
        urllib.request.urlretrieve(urllib.parse.quote(url, safe=":/?&="), path)
        if md5 is not None:
            downloaded_md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
            if md5 == downloaded_md5:
                return
            else:
                logger.warning(f"Reference md5 value ({md5}) is not equal to the downloaded ({downloaded_md5})")
        else:
            return
    raise ValueError(f"Cannot download dataset from {url}, reference md5={md5}")


def download_all_resources(url: str, path: str, resources: list, pwd: T.Optional[bytes] = None) -> None:
    for resource_name, _ in resources:
        file_name, _ = os.path.splitext(resource_name)
        resource_to_path = os.path.join(path, file_name)
        if os.path.exists(resource_to_path):
            continue
        else:
            download(urllib.parse.urljoin(url, resource_name), path)
            archive_path = os.path.join(path, resource_name)
            extract_archive(archive_path, pwd=pwd)
