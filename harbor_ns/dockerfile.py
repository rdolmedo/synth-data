"""Dockerfile parser for harbor_ns.

Parses the subset of Dockerfile syntax that Terminal-Bench tasks use into a list
of stages with typed instructions: line continuations, comments, BuildKit
heredocs (``RUN <<EOF``, ``COPY <<EOF /dst``), global and per-stage ARGs,
``${VAR}``/``${VAR:-default}``/``${VAR:+alt}`` substitution, JSON (exec) and
shell forms, multi-stage builds, and BuildKit flags (``--mount``, ``--network``,
``--platform``, ``--chown``, ``--chmod``, ``--from``, ``--link``), which are
parsed and either honoured or ignored by the builder.

RUN commands are not substituted here: like Docker, the builder passes ARG/ENV
values to the shell as environment variables.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field

_HEREDOC_RE = re.compile(r"<<(-?)\s*([\"']?)([A-Za-z_][A-Za-z0-9_]*)\2")


@dataclass
class Heredoc:
    name: str
    content: str
    strip_tabs: bool = False


@dataclass
class Instruction:
    cmd: str  # upper-case instruction keyword
    raw: str  # argument text (after flags have NOT been removed)
    flags: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)  # parsed args for COPY/ADD etc.
    json_form: bool = False
    heredocs: list[Heredoc] = field(default_factory=list)
    lineno: int = 0


@dataclass
class Stage:
    base: str
    name: str | None
    index: int
    platform: str | None = None
    instructions: list[Instruction] = field(default_factory=list)


class DockerfileError(ValueError):
    pass


def _logical_lines(text: str, escape: str = "\\") -> list[tuple[int, str, list[Heredoc]]]:
    """Split into logical instructions, handling continuations and heredocs."""
    lines = text.split("\n")
    out: list[tuple[int, str, list[Heredoc]]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        start = i + 1
        buf = line
        i += 1
        # Join continuation lines like Docker: drop the escape + newline and
        # append the next line verbatim; empty and comment-only lines inside a
        # continuation are skipped.
        while buf.rstrip().endswith(escape):
            buf = buf.rstrip()[:-1]
            while i < n and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
                i += 1
            if i >= n:
                break
            buf += lines[i]
            i += 1
        heredocs: list[Heredoc] = []
        keyword = buf.strip().split(None, 1)[0].upper() if buf.strip() else ""
        if keyword in ("RUN", "COPY", "ADD"):
            for m in _HEREDOC_RE.finditer(buf):
                strip_tabs = m.group(1) == "-"
                name = m.group(3)
                body: list[str] = []
                while i < n:
                    cand = lines[i]
                    test = cand.lstrip("\t") if strip_tabs else cand
                    i += 1
                    if test.rstrip("\r") == name:
                        break
                    body.append(cand.lstrip("\t") if strip_tabs else cand)
                else:
                    raise DockerfileError(f"line {start}: unterminated heredoc {name}")
                heredocs.append(Heredoc(name=name, content="\n".join(body) + "\n", strip_tabs=strip_tabs))
        out.append((start, buf.strip(), heredocs))
    return out


def substitute(value: str, env: dict[str, str]) -> str:
    """Docker-style variable substitution ($VAR, ${VAR}, ${VAR:-x}, ${VAR:+x}, escapes)."""
    res = []
    i = 0
    n = len(value)
    while i < n:
        c = value[i]
        if c == "\\" and i + 1 < n and value[i + 1] == "$":
            res.append("$")
            i += 2
            continue
        if c == "$" and i + 1 < n:
            if value[i + 1] == "{":
                j = value.find("}", i + 2)
                if j < 0:
                    res.append(value[i:])
                    break
                expr = value[i + 2 : j]
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?::([-+])(.*))?$", expr, re.S)
                if not m:
                    res.append(value[i : j + 1])
                else:
                    name, op, word = m.group(1), m.group(2), m.group(3)
                    cur = env.get(name)
                    if op == "-":
                        res.append(cur if cur else substitute(word or "", env))
                    elif op == "+":
                        res.append(substitute(word or "", env) if cur else "")
                    else:
                        res.append(cur or "")
                i = j + 1
                continue
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", value[i + 1 :])
            if m:
                res.append(env.get(m.group(0), ""))
                i += 1 + len(m.group(0))
                continue
        res.append(c)
        i += 1
    return "".join(res)


def _split_flags(rest: str) -> tuple[dict[str, str], str]:
    """Strip leading --flag=value / --flag tokens."""
    flags: dict[str, str] = {}
    s = rest.lstrip()
    while s.startswith("--"):
        m = re.match(r"--([A-Za-z-]+)(?:=(\"[^\"]*\"|'[^']*'|\S*))?\s*", s)
        if not m:
            break
        val = m.group(2) or ""
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        key = m.group(1)
        flags[key] = (flags[key] + "," + val) if key in flags else val
        s = s[m.end() :]
    return flags, s


def _try_json(rest: str) -> list[str] | None:
    s = rest.strip()
    if s.startswith("["):
        try:
            v = json.loads(s)
        except json.JSONDecodeError:
            return None
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
    return None


def parse_env_pairs(rest: str) -> list[tuple[str, str]]:
    """ENV/LABEL/ARG style ``k=v k2="v 2"`` or legacy ``ENV k v``."""
    s = rest.strip()
    first = s.split(None, 1)
    if first and "=" not in first[0]:
        # legacy form: ENV key value with spaces
        key = first[0]
        val = first[1] if len(first) > 1 else ""
        return [(key, val)]
    lex = shlex.shlex(s, posix=True)
    lex.whitespace_split = True
    lex.commenters = ""
    lex.escape = "\\"
    pairs = []
    for tok in lex:
        if "=" in tok:
            k, v = tok.split("=", 1)
            pairs.append((k, v))
        else:
            pairs.append((tok, ""))
    return pairs


def parse(text: str) -> tuple[list[Stage], dict[str, str]]:
    """Parse a Dockerfile. Returns (stages, global_args)."""
    escape = "\\"
    m = re.match(r"\s*#\s*escape\s*=\s*(\S)", text)
    if m:
        escape = m.group(1)
    stages: list[Stage] = []
    global_args: dict[str, str] = {}
    for lineno, line, heredocs in _logical_lines(text, escape):
        parts = line.split(None, 1)
        cmd = parts[0].upper()
        rest = parts[1] if len(parts) > 1 else ""
        if cmd == "FROM":
            flags, rest2 = _split_flags(rest)
            toks = rest2.split()
            if not toks:
                raise DockerfileError(f"line {lineno}: empty FROM")
            base = substitute(toks[0], global_args)
            name = None
            if len(toks) >= 3 and toks[1].upper() == "AS":
                name = toks[2].lower()
            stages.append(Stage(base=base, name=name, index=len(stages), platform=flags.get("platform")))
            continue
        if not stages:
            if cmd == "ARG":
                for k, v in parse_env_pairs(rest):
                    global_args[k] = v
                continue
            raise DockerfileError(f"line {lineno}: {cmd} before FROM")
        flags, body = ({}, rest)
        if cmd in ("RUN", "COPY", "ADD"):
            flags, body = _split_flags(rest)
        ins = Instruction(cmd=cmd, raw=body, flags=flags, heredocs=heredocs, lineno=lineno)
        jf = _try_json(body)
        if jf is not None and cmd in ("RUN", "CMD", "ENTRYPOINT", "SHELL", "COPY", "ADD", "VOLUME"):
            ins.args = jf
            ins.json_form = True
        elif cmd in ("COPY", "ADD"):
            ins.args = shlex.split(body, posix=True) if not heredocs else body.split()
        stages[-1].instructions.append(ins)
    if not stages:
        raise DockerfileError("no FROM instruction")
    return stages, global_args
