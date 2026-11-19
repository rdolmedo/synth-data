"""Scrub secrets and PII from an asset's text files before it enters the store (BLUEPRINT.md: scrub secrets and PII
from logs and configs). High-confidence secret patterns are redacted; e-mail addresses become stable pseudonyms
under example.com (same address -> same pseudonym, so joins inside the asset still work). Binary files are left as
they are. git repositories: author/committer e-mails are pseudonymized too, and a secret anywhere in history
rejects the asset (history blobs are not rewritten)."""
import hashlib
import re
import subprocess
from pathlib import Path

SECRETS = [
    re.compile(rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----.*?-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----', re.S),
    re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(rb'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b'),
    re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{60,}\b'),
    re.compile(rb'\bxox[baprs]-[A-Za-z0-9-]{10,}\b'),
    re.compile(rb'\bAIza[0-9A-Za-z\-_]{35}\b'),
    re.compile(rb'\bsk_live_[0-9a-zA-Z]{24,}\b'),
    re.compile(rb'\bhf_[A-Za-z0-9]{34,}\b'),
    re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b'),
    re.compile(rb'\bglpat-[A-Za-z0-9\-_]{20}\b'),
    re.compile(rb'https://hooks\.slack\.com/services/[A-Za-z0-9/]{20,}'),
]
EMAIL = re.compile(rb'\b[A-Za-z0-9._%+\-]{1,64}@(?:[A-Za-z0-9\-]{1,63}\.)+[A-Za-z]{2,24}\b')
KEEP_DOMAINS = (b'example.com', b'example.org', b'example.net', b'localhost', b'users.noreply.github.com',
                b'noreply.github.com', b'test.com', b'domain.com', b'email.com')
TEXT_MAX = 8 << 20


def pseudonym(addr: bytes) -> bytes:
    return b'user' + hashlib.sha256(addr.lower()).hexdigest()[:8].encode() + b'@example.com'


def is_text(data: bytes) -> bool:
    if b'\0' in data[:8192]:
        return False
    try:
        data[:8192].decode('utf-8')
        return True
    except UnicodeDecodeError as e:
        return e.start > 8180  # a multi-byte character cut at the boundary


def scrub_bytes(data: bytes):
    n_secret = n_email = 0
    for pat in SECRETS:
        data, k = pat.subn(b'REDACTED_SECRET', data)
        n_secret += k

    def mail(m):
        nonlocal n_email
        addr = m.group(0)
        dom = addr.rsplit(b'@', 1)[1].lower()
        if dom.endswith(KEEP_DOMAINS) or re.search(rb'\.(png|jpe?g|gif|svg|webp)$', addr, re.I):
            return addr
        n_email += 1
        return pseudonym(addr)
    data = EMAIL.sub(mail, data)
    return data, n_secret, n_email


def scrub_tree(root):
    """In place. -> {'files_scrubbed', 'secrets', 'emails'}."""
    root = Path(root)
    out = {'files_scrubbed': 0, 'secrets': 0, 'emails': 0}
    for p in root.rglob('*'):
        if p.is_symlink() or not p.is_file() or '.git' in p.relative_to(root).parts:
            continue
        if p.stat().st_size > TEXT_MAX:
            continue
        data = p.read_bytes()
        if not is_text(data):
            continue
        new, s, e = scrub_bytes(data)
        if new != data:
            p.write_bytes(new)
            out['files_scrubbed'] += 1
            out['secrets'] += s
            out['emails'] += e
    return out


def history_has_secret(repo):
    """Any secret pattern in any blob of a git repository's history."""
    r = subprocess.run(['git', '-C', str(repo), 'log', '--all', '-p', '--no-color', '--no-ext-diff'],
                       capture_output=True, timeout=600)
    return any(p.search(r.stdout) for p in SECRETS)


MAILMAP_FILTER = r'''
pseudo() { printf 'user%s@example.com' "$(printf '%s' "$1" | tr 'A-Z' 'a-z' | sha256sum | cut -c1-8)"; }
export GIT_AUTHOR_EMAIL="$(pseudo "$GIT_AUTHOR_EMAIL")"
export GIT_COMMITTER_EMAIL="$(pseudo "$GIT_COMMITTER_EMAIL")"
'''


def pseudonymize_git(repo):
    """Rewrite author/committer e-mails of every commit (names kept: they are the attribution)."""
    env = {'FILTER_BRANCH_SQUELCH_WARNING': '1', 'PATH': '/usr/bin:/bin', 'HOME': str(repo)}
    r = subprocess.run(['git', '-C', str(repo), 'filter-branch', '-f', '--env-filter', MAILMAP_FILTER, '--', '--all'],
                       capture_output=True, text=True, env=env, timeout=1800)
    if r.returncode:
        raise RuntimeError(f'git filter-branch failed: {r.stderr[-500:]}')
    subprocess.run(['rm', '-rf', str(Path(repo) / '.git' / 'refs' / 'original')], check=True)
    subprocess.run(['git', '-C', str(repo), 'reflog', 'expire', '--expire=now', '--all'], capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'gc', '--prune=now', '-q'], capture_output=True)
