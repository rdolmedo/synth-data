"""JSON Lines readers: records are separated by '\\n' only (#348).

`str.splitlines()` also splits on \\x0b \\x0c \\x1c-\\x1e \\x85 U+2028 U+2029, which JSON allows raw inside strings
(JavaScript's JSON.stringify, e.g. Pi's session files, and Python's json.dumps(ensure_ascii=False) leave them
unescaped): a record holding one is cut in two and neither half decodes. These readers split on '\\n' only (a '\\r'
before it is a CRLF line ending).

A final line without its newline is a record a concurrent writer is still appending: it is kept if it decodes
(a writer that omits the last newline) and skipped otherwise (read again next time). Any other line that does not
decode raises JSONLError naming the file and line, unless the caller passes `bad`, a list that collects
(path, line number, reason) so the caller can count them and mark its result invalid. Nothing is dropped silently.
"""
from __future__ import annotations

import json
from pathlib import Path


class JSONLError(ValueError):
    pass


def lines(text: str) -> list[str]:
    """The physical lines of a JSONL text: split on '\\n' only, a trailing '\\r' removed, the last line included
    even if it has no newline ('' when the text ends with one)."""
    return [x.removesuffix('\r') for x in text.split('\n')]


def loads(text: str, *, where: str = '<text>', bad: list | None = None) -> list:
    parts = lines(text)
    tail = parts.pop()
    out = []
    for i, line in enumerate(parts, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError as e:
            if bad is None:
                raise JSONLError(f'{where}:{i}: undecodable JSON line ({e})') from None
            bad.append((where, i, str(e)))
    if tail.strip():
        try:
            out.append(json.loads(tail))
        except ValueError:
            pass   # a record still being appended
    return out


def read(path, *, bad: list | None = None, required: bool = False) -> list:
    """The records of a JSONL file ([] if it does not exist, FileNotFoundError if `required`); see the module
    docstring for bad lines."""
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return []
    return loads(p.read_text(encoding='utf-8'), where=str(p), bad=bad)


def stream(f, *, where: str = '<stream>', bad: list | None = None):
    """Records of an open text file object read in full (e.g. gzip.open(..., 'rt')), same rules as loads()."""
    return loads(f.read(), where=where, bad=bad)


class Tolerant:
    """For a run's own append-only files, whose readers must keep going past a bad line (a torn line of a killed
    job, followed by later appends): each bad line is recorded per file (and warned once on stderr) so the run's
    summary can count it and fail its validity instead of losing the row silently."""

    def __init__(self):
        self.bad: dict[str, dict[int, str]] = {}

    def read(self, path) -> list:
        import sys
        got: list = []
        rows = read(path, bad=got)
        for where, i, why in got:
            seen = self.bad.setdefault(where, {})
            if i not in seen:
                seen[i] = why
                print(f'jsonl: {where}:{i}: undecodable line skipped ({why})', file=sys.stderr, flush=True)
        return rows

    def count(self, under) -> int:
        """Bad lines seen so far in files under the directory `under`."""
        root = str(Path(under).resolve())
        return sum(len(v) for k, v in self.bad.items() if str(Path(k).resolve()).startswith(root + '/'))
