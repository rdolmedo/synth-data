"""JSONL logs: whole-line appends and torn-line-tolerant reads (#451 workstream 2, lesson "torn wire writes").

M5, 11-15: 8 TB-Sci-lite second traces had one response-budget wire line holding a cut fragment followed by a whole
record (5 of 8 ended 'error'), and the HLE and TB-Sci-lite resample builds each crashed once on such a line. A record
of a long session is MBs. A write(2) cut short (a full or over-quota filesystem, a signal) leaves part of a record
without its newline, and the next append glues its record onto that fragment.

Writers: `append_line(path, json.dumps(record, ...))` sends the whole line in one write(2) on an O_APPEND descriptor
(appends from several processes never interleave), continues a short write, and if the write fails after part of
the line went out it ends the fragment with a newline (best effort) before raising, so the NEXT record starts on its
own line instead of being glued to the fragment. The bytes of a successful append are exactly `line + '\\n'`, as
the text-mode appends it replaces (CPython already issued one write(2) for one f.write(line + '\\n'); what it did
not do is continue a short write or terminate a cut one).

Readers: `read(path)` / `parse(data)` return the records plus what was repaired:
  - a line holding a fragment followed by a whole record keeps the record (`recovered`): the record starts with the
    same first key as every record of its writer ('{"time":', '{"t":'), so it is found at the last occurrence of the
    line's first-key signature after offset 0 that parses to the end of the line and carries every key the file's
    clean records all have (a nested object with the same first key is not taken for the record);
  - a line that starts like the file's records but neither parses nor recovers is one cut record (`cut`): the
    fragment a failed write ended with a newline, the same loss as a fragment glued to the next record;
  - a last line without its newline that neither parses nor recovers is a cut record: skipped, `cut_tail` set;
  - any other unreadable line is listed in `bad` (1-based line numbers): the caller decides (raise, or count a drop).
`recovered` / `cut` / `cut_tail` each mean one record was lost; whether that matters is the caller's call (a notice
map must not be used with a lost entry; a response-budget wire record is self-contained).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os

MAX_CANDIDATES = 64   # record starts tried per torn line (each try parses at most the line's length)


def append_line(path, line: str) -> None:
    """Append `line` (a newline is added) to `path` in one write(2) on an O_APPEND descriptor."""
    data = (line if line.endswith('\n') else line + '\n').encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        view, sent = memoryview(data), 0
        try:
            while sent < len(data):   # a regular file takes it all at once; a short write (signal, quota) continues
                sent += os.write(fd, view[sent:])
        except OSError:
            if sent:   # end the fragment so the next record does not glue onto it
                try:
                    os.write(fd, b'\n')
                except OSError:
                    pass
            raise
    finally:
        os.close(fd)


@dataclass
class Lines:
    records: list = field(default_factory=list)
    cut_tail: bool = False      # the last line had no newline and could not be read (a killed writer): skipped
    recovered: int = 0          # torn lines whose trailing whole record was kept
    cut: int = 0                # lines holding only a cut record of the file's writer (ended by a newline): skipped
    bad: list = field(default_factory=list)   # 1-based numbers of other unreadable lines (skipped)

    @property
    def clean(self) -> bool:
        return not (self.cut_tail or self.recovered or self.cut or self.bad)


def _signature(line: bytes) -> bytes | None:
    """b'{"<first key>":' of a record line (the object whose first key ends at the line's first '":')."""
    colon = line.find(b'":')
    if colon < 0:
        return None
    first = line.rfind(b'{', 0, colon)
    sig = line[first:colon + 2] if first >= 0 else b''
    return sig if sig.startswith(b'{"') else None


def recover(line: bytes, keys=frozenset()):
    """The whole record at the end of a torn line (a fragment, then a record of the same writer), or None. keys: the
    record must have all of them (the keys every clean record of the file has)."""
    if not line.startswith(b'{'):
        return None
    signature = _signature(line)         # e.g. b'{"time":' (a fragment shorter than the first key may precede it)
    if signature is None:
        return None
    end = len(line)
    for _ in range(MAX_CANDIDATES):
        j = line.rfind(signature, 1, end)
        if j <= 0:
            return None
        try:
            value = json.loads(line[j:])
        except ValueError:
            end = j
            continue
        if isinstance(value, dict) and keys <= value.keys():
            return value
        end = j
    return None


def parse(data: bytes) -> Lines:
    out = Lines()
    lines = data.split(b'\n')
    complete = data.endswith(b'\n')
    if complete:
        lines.pop()            # the empty string after the final newline
    parsed, keys, sig = [], None, None
    for line in lines:         # pass 1: the clean records, their common keys and first-key signature
        try:
            value = json.loads(line) if line.strip() else None
        except ValueError:
            value = ...
        parsed.append(value)
        if isinstance(value, dict):
            keys = set(value) if keys is None else keys & set(value)
            sig = sig or _signature(line)
    keys = frozenset(keys or ())
    for number, (line, value) in enumerate(zip(lines, parsed), 1):
        if value is None:
            continue
        if value is not ...:
            out.records.append(value)
            continue
        value = recover(line, keys)
        if value is not None:
            out.records.append(value)
            out.recovered += 1
        elif number == len(lines) and not complete:
            out.cut_tail = True
        elif sig and (line.startswith(sig) or sig.startswith(line)):
            out.cut += 1
        else:
            out.bad.append(number)
    return out


def read(path) -> Lines:
    with open(path, 'rb') as f:
        return parse(f.read())
