"""Local file storage for uploaded documents.

Stored filenames are generated server-side (``<uuid>.<ext>``); client filenames are
never used to build filesystem paths.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

_STORED_NAME = re.compile(r"^[0-9a-f]{32}\.(pdf|txt)$")


class FileStorage:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, stored_name: str) -> Path:
        if not _STORED_NAME.match(stored_name):
            raise ValueError("Invalid stored filename")
        path = (self.root / stored_name).resolve()
        if path.parent != self.root.resolve():
            raise ValueError("Path escapes storage root")
        return path

    def save(self, data: bytes, kind: str) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex}.{'pdf' if kind == 'pdf' else 'txt'}"
        path = self._path(stored_name)
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return stored_name

    def read(self, stored_name: str) -> bytes:
        return self._path(stored_name).read_bytes()

    def exists(self, stored_name: str) -> bool:
        try:
            return self._path(stored_name).exists()
        except ValueError:
            return False

    def delete(self, stored_name: str) -> None:
        try:
            self._path(stored_name).unlink(missing_ok=True)
        except ValueError:
            pass
