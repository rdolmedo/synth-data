"""Real binaries from Ubuntu 24.04 (noble) binary packages: ELF executables (format elf-executable), object files
from static libraries (object-file), Java bytecode (bytecode) and firmware blobs (firmware-rom). Noble matches the
task base image (local/tasksynth-base, Ubuntu 24.04), so the executables run in the sandbox.

Search reads the archive's Packages indexes (main + universe, cached once per process in node-local scratch) and
scores package descriptions against the query, preferring small, obscure packages over famous tools and anything the
base image already installs. Fetch downloads the .deb at the index's pinned version, verifies its SHA-256, unpacks
it and keeps the binaries plus the package's copyright file; the license comes from that copyright file.
"""
import asyncio
import json
import lzma
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..http import TooLarge
from ..license import classify
from .base import Reject, Source, cap

MIRROR = 'http://archive.ubuntu.com/ubuntu'
DIST = 'noble'
COMPONENTS = ('main', 'universe')
CACHE = Path(os.environ.get('TMPDIR', '/tmp')) / 'tasksynth-debian-index'
BASE_MANIFEST = Path('/data/aai/tbench/harbor_ns/bases/local_tasksynth-base_2.manifest.json')
FAMOUS = {'coreutils', 'grep', 'sed', 'gawk', 'mawk', 'curl', 'wget', 'bash', 'tar', 'gzip', 'bzip2', 'xz-utils',
          'openssh-client', 'openssh-server', 'vim', 'nano', 'git', 'perl', 'python3', 'gcc', 'g++', 'make',
          'busybox', 'busybox-static', 'util-linux', 'findutils', 'diffutils', 'less', 'procps', 'htop', 'tmux',
          'screen', 'zip', 'unzip', 'jq', 'sqlite3', 'nginx', 'apache2', 'netcat-openbsd', 'nmap', 'tcpdump',
          'strace', 'gdb', 'binutils', 'file', 'openssl', 'rsync', 'zsh', 'emacs', 'ffmpeg', 'imagemagick'}
STOP = set('the a an of for and or to in on with from by at as is are be this that into using use tool tools data '
           'file files program programs utility utilities linux command line based simple small real set'.split())
# sections that hold user-facing programs (elf-executable)
PROGRAM_SECTIONS = {'utils', 'net', 'admin', 'science', 'math', 'text', 'misc', 'devel', 'editors', 'games', 'sound',
                    'video', 'graphics', 'electronics', 'hamradio', 'database', 'mail', 'web', 'comm', 'otherosfs',
                    'shells', 'vcs', 'httpd', 'gnu-r', 'ocaml', 'interpreters', 'embedded', 'news', 'x11'}
KEEP_PREFIXES = {
    'elf-executable': ('usr/bin/', 'usr/sbin/', 'bin/', 'sbin/', 'usr/games/', 'usr/libexec/'),
    'object-file': ('usr/lib/',),
    'bytecode': ('usr/share/java/',),
    'firmware-rom': ('lib/firmware/', 'usr/lib/firmware/', 'usr/share/firmware/', 'usr/share/', 'usr/lib/'),
}
DEP5_MAP = {'gpl-2+': 'gpl-2.0-or-later', 'gpl-2': 'gpl-2.0-only', 'gpl-2.0+': 'gpl-2.0-or-later',
            'gpl-3+': 'gpl-3.0-or-later', 'gpl-3': 'gpl-3.0-only', 'gpl-3.0+': 'gpl-3.0-or-later',
            'lgpl-2+': 'lgpl-2.1-or-later', 'lgpl-2.1+': 'lgpl-2.1-or-later', 'lgpl-2.1': 'lgpl-2.1-only',
            'lgpl-3+': 'lgpl-3.0-or-later', 'lgpl-3': 'lgpl-3.0-only', 'agpl-3+': 'agpl-3.0-or-later',
            'agpl-3': 'agpl-3.0-only', 'expat': 'mit', 'mit': 'mit', 'bsd-2-clause': 'bsd-2-clause',
            'bsd-3-clause': 'bsd-3-clause', 'bsd-4-clause': 'bsd-4-clause', 'isc': 'isc', 'apache-2.0': 'apache-2.0',
            'apache-2': 'apache-2.0', 'zlib': 'zlib', 'public-domain': 'public-domain', 'mpl-2.0': 'mpl-2.0',
            'artistic': 'artistic-2.0', 'artistic-2.0': 'artistic-2.0', 'boost-1.0': 'bsl-1.0', 'bsl-1.0': 'bsl-1.0',
            'cc0-1.0': 'cc0-1.0', 'cc0': 'cc0-1.0', 'unlicense': 'unlicense', 'wtfpl': 'wtfpl', 'x11': 'x11',
            'curl': 'curl', 'mpl-1.1': 'mpl-1.1', 'epl-1.0': 'epl-1.0', 'epl-2.0': 'epl-2.0', 'bsd': 'bsd-3-clause',
            'gpl': 'gpl-2.0-or-later', 'lgpl': 'lgpl-2.1-or-later', 'gpl-1+': 'gpl-2.0-or-later', 'ofl-1.1': 'ofl-1.1',
            'python-2.0': 'python-2.0', 'psf-2': 'python-2.0', 'zpl-2.1': 'zpl-2.1', 'ncsa': 'ncsa',
            'lgpl-2.0+': 'lgpl-2.1-or-later', 'lgpl-2': 'lgpl-2.1-only', 'lgpl-2.0': 'lgpl-2.1-only',
            'lgpl-3.0+': 'lgpl-3.0-or-later', 'gpl-2.0': 'gpl-2.0-only', 'gpl-3.0': 'gpl-3.0-only',
            'gpl-2.0-or-later': 'gpl-2.0-or-later', 'gpl-3.0-or-later': 'gpl-3.0-or-later', 'bsd-3': 'bsd-3-clause',
            'bsd-2': 'bsd-2-clause', 'mit/x11': 'mit', 'permissive': None}
GUI_DEPENDS = ('libgtk', 'libqt', 'libkf5', 'libkf6', 'qml-module', 'libwx', 'libgnome', 'libsdl', 'libgl1',
               'libx11-6', 'libwebkit', 'libadwaita')
LANG_PACKAGING = ('ocaml', 'golang', 'rust-', 'haskell', 'ghc', 'node-', 'perl', 'ruby', 'python', 'php', 'java',
                  'r-cran', 'elpa-', 'cl-', 'gir1.2')


def parse_packages(text, component):
    out = []
    for para in text.split('\n\n'):
        if not para.strip():
            continue
        d, key = {}, None
        for line in para.split('\n'):
            if line.startswith((' ', '\t')) and key:
                d[key] += '\n' + line.strip()
            elif ':' in line:
                key, _, v = line.partition(':')
                d[key] = v.strip()
        if 'Package' in d and 'Filename' in d:
            d['Component'] = component
            out.append(d)
    return out


def dep5_license(text):
    """License of the 'Files: *' paragraph of a DEP-5 copyright file (the first allowed alternative of "A or B"), or
    a free-text guess for non-DEP-5 files. Other paragraphs (debian/*, single files) never decide it.
    -> raw id for license.classify, or None."""
    paras = text.split('\n\n')
    star = [p for p in paras if re.search(r'^Files:\s*\*\s*$', p, re.M)]
    dep5 = bool(re.search(r'^Format:.*copyright-format', text, re.M)) or any(re.search(r'^Files:', p, re.M) for p in paras)
    scope = star if star else ([] if dep5 else paras)
    for p in scope:
        m = re.search(r'^License:\s*(.+)$', p, re.M)
        if not m:
            continue
        for alt in re.split(r'\s+or\s+|\s*\|\s*', m.group(1).strip(), flags=re.I):
            alt = alt.strip().lower().rstrip(',')
            alt = re.sub(r'\s+with\s+.*$', '', alt)  # "GPL-2+ with OpenSSL exception"
            alt = re.sub(r'[-\s]*(?:with[-\s].*)?exception$', '', alt)  # "LGPL-2.1-exception"
            spdx = DEP5_MAP.get(alt, alt)
            if classify(spdx)[1] != 'excluded':
                return spdx
        body = p[m.end():]
        if body.strip():
            return free_text_license(body) or None
        return None
    if dep5 and not star:
        return None
    return free_text_license(text)


def free_text_license(text):
    t = text.lower()
    if 'gnu general public license' in t:
        later = 'any later version' in t
        if 'version 3' in t:
            return 'gpl-3.0-or-later' if later else 'gpl-3.0-only'
        return 'gpl-2.0-or-later' if later else 'gpl-2.0-only'
    if 'gnu lesser general public' in t or 'gnu library general public' in t:
        return 'lgpl-2.1-or-later'
    if 'apache license' in t and '2.0' in t:
        return 'apache-2.0'
    if 'permission is hereby granted, free of charge' in t:
        return 'mit'
    if 'redistribution and use in source and binary forms' in t:
        return 'bsd-3-clause' if 'neither the name' in t else 'bsd-2-clause'
    if 'public domain' in t:
        return 'public-domain'
    return None


class DebianBinaries(Source):
    name = 'debian'
    classes = ('binary',)
    _index = None
    _lock = None

    async def index(self):
        if DebianBinaries._index is not None:
            return DebianBinaries._index
        if DebianBinaries._lock is None:
            DebianBinaries._lock = asyncio.Lock()
        async with DebianBinaries._lock:
            if DebianBinaries._index is not None:
                return DebianBinaries._index
            CACHE.mkdir(parents=True, exist_ok=True)
            pkgs = []
            for comp in COMPONENTS:
                f = CACHE / f'{DIST}-{comp}-Packages.xz'
                if not f.exists():
                    await self.http.download(f'{MIRROR}/dists/{DIST}/{comp}/binary-amd64/Packages.xz',
                                             f.with_suffix('.partial'), 200 << 20)
                    f.with_suffix('.partial').replace(f)
                pkgs += parse_packages(lzma.decompress(f.read_bytes()).decode('utf-8', 'replace'), comp)
            try:
                base = set(json.loads(BASE_MANIFEST.read_text()).get('apt', []))
            except (OSError, ValueError):
                base = set()
            DebianBinaries._index = [p for p in pkgs if p['Package'] not in FAMOUS and p['Package'] not in base]
            return DebianBinaries._index

    @staticmethod
    def eligible(p, fmt):
        name, section = p['Package'], p.get('Section', '').split('/')[-1]
        size = int(p.get('Installed-Size') or 0)
        depends = p.get('Depends', '')
        if fmt == 'elf-executable':
            return (section in PROGRAM_SECTIONS and size < 5000 and not name.endswith(('-dev', '-doc', '-data', '-dbg'))
                    and not name.startswith(('lib', 'python3-', 'ruby-', 'node-', 'golang-', 'r-cran-', 'fonts-'))
                    and not any(g in depends for g in GUI_DEPENDS) and 'libc6' in depends)
        if fmt == 'object-file':
            return (name.endswith('-dev') and section in ('libdevel', 'devel') and size < 8000
                    and not any(x in name for x in LANG_PACKAGING))
        if fmt == 'bytecode':
            return section == 'java' and name.endswith('-java') and size < 6000
        if fmt == 'firmware-rom':
            return ('firmware' in name or 'rom' in name.split('-')) and size < 40000 and section in (
                'misc', 'kernel', 'otherosfs', 'electronics', 'admin', 'games', 'utils', 'embedded')
        return False

    async def search(self, target, query, n):
        fmts = [f for f in (target.get('formats') or ['elf-executable']) if f in KEEP_PREFIXES] or ['elf-executable']
        fmt = random.choice(fmts)
        words = [w for w in re.findall(r'[a-z0-9][a-z0-9+\-]{2,}', query.lower()) if w not in STOP]
        if not words:
            return []
        pool = [p for p in await self.index() if self.eligible(p, fmt)]
        hay = {id(p): f'{p["Package"]} {p.get("Description", "").lower()} {p.get("Tag", "")}' for p in pool}
        # rare query words decide (IDF): "astronomy ephemeris calculator" must not match every calculator
        df = {w: sum(1 for p in pool if w in hay[id(p)]) for w in words}
        idf = {w: math.log((len(pool) + 1) / (df[w] + 1)) for w in words}
        need = max(idf.values(), default=0) * 0.6
        scored = []
        for p in pool:
            h = hay[id(p)]
            hit = [w for w in words if w in h]
            score = sum(idf[w] for w in hit)
            if not hit or score < need:
                continue
            desc = h
            score += 1.0 if re.search(r'command[ -]line|\bcli\b|console', desc) else 0.0
            score += 0.5 if any(w in p['Package'] for w in hit) else 0.0
            score -= int(p.get('Installed-Size') or 0) / 5000
            scored.append((score + random.random() * 0.5, p))
        scored.sort(key=lambda x: -x[0])
        out = []
        for _, p in scored[: n * 3]:
            desc = p.get('Description', '').replace('\n', ' ')
            out.append({'source': self.name, 'key': f'{p["Package"]}={p["Version"]}',
                        'url': f'{MIRROR}/{p["Filename"]}',
                        'page_url': f'https://launchpad.net/ubuntu/{DIST}/+package/{p["Package"]}',
                        'title': f'{p["Package"]} {p["Version"]} (Ubuntu {DIST} {p["Component"]}, {p.get("Section")})',
                        'description': desc[:400], 'license_raw': None, 'size': int(p.get('Size') or 0),
                        'formats': [fmt],
                        'meta': {k: p.get(k) for k in ('Package', 'Version', 'Filename', 'SHA256', 'Size',
                                                         'Installed-Size', 'Maintainer', 'Source', 'Homepage',
                                                         'Section', 'Component')} | {'format': fmt},
                        'query': query})
        return out

    async def fetch(self, cand, staging, target):
        m = cand['meta']
        staging = Path(staging)
        with tempfile.TemporaryDirectory(prefix='deb-', dir=staging.parent) as tmp:
            deb = Path(tmp) / Path(m['Filename']).name
            try:
                digest, nbytes = await self.http.download(cand['url'], deb, cap(target))
            except TooLarge as e:
                raise Reject(str(e))
            if m.get('SHA256') and digest != m['SHA256']:
                raise Reject(f'SHA-256 mismatch for {deb.name}: {digest} vs index {m["SHA256"]}')
            root = Path(tmp) / 'root'
            root.mkdir()
            if shutil.which('dpkg-deb'):
                r = subprocess.run(['dpkg-deb', '-x', str(deb), str(root)], capture_output=True, text=True)
            else:
                r = subprocess.run(f'cd {tmp} && ar x {deb.name} && tar -xf data.tar.* -C root', shell=True,
                                   capture_output=True, text=True)
            if r.returncode:
                raise Reject(f'unpack failed: {r.stderr[-300:]}')
            fmt = m.get('format') or 'elf-executable'
            prefixes = KEEP_PREFIXES[fmt]
            pkg = m['Package']
            copyright_file = root / 'usr' / 'share' / 'doc' / pkg / 'copyright'
            kept = 0
            for p in sorted(root.rglob('*')):
                if not p.is_file() or p.is_symlink():
                    continue
                rel = p.relative_to(root).as_posix()
                if not rel.startswith(prefixes) or '/doc/' in rel or '/man/' in rel:
                    continue
                if fmt == 'object-file' and not rel.endswith(('.a', '.o')):
                    continue
                if fmt == 'bytecode' and not rel.endswith(('.jar', '.class')):
                    continue
                if fmt == 'elf-executable' and p.read_bytes()[:4] != b'\x7fELF':
                    continue
                dest = staging / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                kept += 1
            if not kept:
                raise Reject(f'{pkg}: no {fmt} files in the package')
            lic_text = copyright_file.read_text(errors='replace') if copyright_file.exists() else ''
            if lic_text:
                d = staging / 'usr' / 'share' / 'doc' / pkg
                d.mkdir(parents=True, exist_ok=True)
                (d / 'copyright').write_text(lic_text)
            license_raw = dep5_license(lic_text) if lic_text else None
        if classify(license_raw)[1] == 'excluded':
            raise Reject(f'{pkg}: license not recognized or not allowed ({license_raw})')
        maint = re.sub(r'\s*<[^>]*>', '', m.get('Maintainer') or '').strip()
        return {'version': f'{pkg}_{m["Version"]} sha256:{digest}', 'sha256': digest, 'download_bytes': nbytes,
                'url': cand['url'], 'license_raw': license_raw,
                'attribution': f'{pkg} {m["Version"]} (Ubuntu {DIST} {m.get("Component")}), maintainer '
                               f'{maint}, upstream source {m.get("Source") or pkg}'
                               + (f' ({m["Homepage"]})' if m.get('Homepage') else '') + f', {license_raw}'}
