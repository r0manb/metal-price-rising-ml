import os
import pathlib
from typing import Union


def get_absolute_path(path: Union[str, os.PathLike]):
    if isinstance(path, str):
        path = path.lstrip("/")

    return pathlib.Path(path).resolve()
