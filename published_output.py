"""Read the public, scheduled NHL outputs without depending on an app reboot."""

from __future__ import annotations

import os
import re
import tempfile
from hashlib import sha1
from pathlib import Path
from urllib.parse import quote

import requests


REPOSITORY = "spillja89/Warlord-NHL-Prop-Tool"
BRANCH = "launch/2026-warlord-readiness"
RAW_BASE = f"https://raw.githubusercontent.com/{REPOSITORY}/{BRANCH}/output"
GRADE_API = f"https://api.github.com/repos/{REPOSITORY}/contents/output/graded"
GRADE_NAME = re.compile(r"(?:tracker_\d{4}-\d{2}-\d{2}_(?:summary\.json|GRADED\.csv)|moves_\d{4}-\d{2}-\d{2}\.csv)\Z")


def fetch_tracker(*, session=requests) -> bytes:
    response = session.get(f"{RAW_BASE}/tracker_latest.csv", timeout=12)
    response.raise_for_status()
    content = response.content
    if not content or len(content) > 25 * 1024 * 1024:
        raise ValueError("Published tracker is empty or too large")
    header = content.splitlines()[0].decode("utf-8-sig", errors="replace")
    if "Player" not in header or "Date" not in header:
        raise ValueError("Published tracker has an unexpected header")
    return content


def list_grade_files(*, session=requests) -> list[tuple[str, str]]:
    response = session.get(GRADE_API, params={"ref": BRANCH}, timeout=12)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    listing = response.json()
    if not isinstance(listing, list):
        raise ValueError("Published grade listing is invalid")
    files = []
    for item in listing:
        name = str(item.get("name", ""))
        sha = str(item.get("sha", ""))
        if item.get("type") == "file" and GRADE_NAME.fullmatch(name) and re.fullmatch(r"[0-9a-f]{40}", sha):
            files.append((name, sha))
    return files


def fetch_grade_file(name: str, *, session=requests) -> bytes:
    if not GRADE_NAME.fullmatch(name):
        raise ValueError("Invalid published grade name")
    response = session.get(f"{RAW_BASE}/graded/{quote(name)}", timeout=12)
    response.raise_for_status()
    content = response.content
    if not content or len(content) > 25 * 1024 * 1024:
        raise ValueError("Published grade is empty or too large")
    return content


def save_if_changed(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return False
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def local_matches_sha(path: Path, git_sha: str) -> bool:
    if not path.is_file() or not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        return False
    try:
        content = path.read_bytes()
    except OSError:
        return False
    prefix = f"blob {len(content)}\0".encode("ascii")
    return sha1(prefix + content).hexdigest() == git_sha
