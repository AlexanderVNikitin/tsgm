import os
import zipfile
import typing
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


def _extract_zip(from_path: str, to_path: str, pwd: typing.Optional[bytes]) -> None:
    with zipfile.ZipFile(from_path, "r", compression=zipfile.ZIP_STORED) as zip:
        zip.extractall(to_path, pwd=pwd)


def _extract_targz(from_path: str, to_path: str, pwd: typing.Optional[bytes] = None) -> None:
    try:
        # Open the tar.gz file
        with tarfile.open(from_path, "r:gz") as tar:
            # Extract all the files in the archive
            tar.extractall(to_path)
            logger.info("Files extracted successfully.")
    except tarfile.TarError as e:
        logger.error(f"Failed to extract tar.gz file: {e}")


EXTRACTORS = {
    #  TODO add ".tar" & ".bz2"
    ".zip": _extract_zip,
    ".gz": _extract_targz,
}


def extract_archive(from_path: str, to_path: typing.Optional[str] = None, pwd: typing.Optional[bytes] = None) -> None:
    ext = _archive_type(from_path)
    extractor = EXTRACTORS[ext]

    if to_path is None:
        to_path, _ = os.path.splitext(from_path)
        to_path = os.path.dirname(to_path)

    extractor(from_path, to_path, pwd=pwd)
    os.remove(from_path)


def download(url: str, path: str, md5: typing.Optional[str] = None, max_attempt: int = 3) -> None:
    logger.info(f"### Downloading from {url} ###")
    os.makedirs(path, exist_ok=True)
    resource_name = url.split("/")[-1]
    path = os.path.join(path, resource_name)
    for attempt in range(max_attempt):
        logger.info(f"Attempt {attempt + 1} / {max_attempt}")
        urllib.request.urlretrieve(urllib.parse.quote(url, safe=":/"), path)
        if md5 is not None:
            downloaded_md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
            if md5 == downloaded_md5:
                return
            else:
                logger.warning(f"Reference md5 value ({md5}) is not equal to the downloaded ({downloaded_md5})")
        else:
            return
    raise ValueError(f"Cannot download dataset from {url}, reference md5={md5}")


def download_all_resources(url: str, path: str, resources: list, pwd: typing.Optional[bytes] = None) -> None:
    for resource_name, _ in resources:
        file_name, _ = os.path.splitext(resource_name)
        fnames = [os.path.splitext(fname)[0] for fname in os.listdir(path)]
        if fnames.count(file_name) > 0:
            continue
        else:
            download(urllib.parse.urljoin(url, resource_name), path)
            archive_path = os.path.join(path, resource_name)
            extract_archive(archive_path, pwd=pwd)
