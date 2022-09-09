import os
import zipfile
import typing
import hashlib
import logging
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


EXTRACTORS = {
    #  TODO add ".tar" & ".bz2"
    ".zip": _extract_zip,
}


def extract_archive(from_path: str, to_path: typing.Optional[str] = None, pwd: typing.Optional[bytes] = None) -> None:
    ext = _archive_type(from_path)
    extractor = EXTRACTORS[ext]

    if to_path is None:
        to_path, _ = os.path.splitext(from_path)
        to_path = os.path.dirname(to_path)

    extractor(from_path, to_path, pwd=pwd)
    os.remove(from_path)


def download(url: str, path: str, md5: typing.Optional[str] = None, max_attempt: int = 2) -> None:
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
        resource_to_path = os.path.join(path, file_name)
        if os.path.exists(resource_to_path):
            continue
        else:
            download(urllib.parse.urljoin(url, resource_name), path)
            archive_path = os.path.join(path, resource_name)
            extract_archive(archive_path, pwd=pwd)
