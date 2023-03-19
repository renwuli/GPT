import numpy as np
import zipfile
import os
import os.path as osp
import urllib

__all__ = ["download_url", "extract_zip"]


def download_url(url, folder, log=True):
    """
    Borrowed from "https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/download.py"
    Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print("Using exist file", filename)
        return path
    else:
        if log:
            print("Downloading", url)

    data = urllib.request.urlopen(url)

    with open(path, "wb") as f:
        f.write(data.read())

    return path


def extract_zip(path, folder, log=True):
    """
    Borrowed from "https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/extract.py"
    Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)