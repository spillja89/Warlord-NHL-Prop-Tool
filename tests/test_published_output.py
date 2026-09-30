from hashlib import sha1
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from published_output import (
    fetch_grade_file,
    fetch_tracker,
    list_grade_files,
    local_matches_sha,
    save_if_changed,
)


class Response:
    def __init__(self, content=b"", listing=None, status_code=200):
        self.content = content
        self._listing = listing
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")

    def json(self):
        return self._listing


class Session:
    def __init__(self, response):
        self.response = response
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.response


class PublishedOutputTests(TestCase):
    def test_published_tracker_can_replace_old_snapshot(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "output" / "tracker_latest.csv"
            path.parent.mkdir()
            path.write_bytes(b"Date,Player\nold,old\n")
            session = Session(Response(b"Date,Player\n2026-09-30,New Player\n"))

            content = fetch_tracker(session=session)
            self.assertTrue(save_if_changed(path, content))
            self.assertIn(b"New Player", path.read_bytes())
            self.assertFalse(save_if_changed(path, content))

    def test_only_expected_graded_files_are_accepted(self):
        with TemporaryDirectory() as directory:
            content = b"Player,Outcome\nA,W\n"
            git_sha = sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            session = Session(Response(listing=[
                {"type": "file", "name": "moves_2026-09-30.csv", "sha": git_sha},
                {"type": "file", "name": "../secrets.toml", "sha": git_sha},
            ]))
            self.assertEqual(list_grade_files(session=session), [("moves_2026-09-30.csv", git_sha)])

            path = Path(directory) / "moves_2026-09-30.csv"
            save_if_changed(path, fetch_grade_file(path.name, session=Session(Response(content))))
            self.assertTrue(local_matches_sha(path, git_sha))
            with self.assertRaises(ValueError):
                fetch_grade_file("../secrets.toml", session=session)
