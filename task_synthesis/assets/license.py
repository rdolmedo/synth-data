"""Licensing policy (BLUEPRINT.md, decided 2026-10-30): one lane, release everything. Any license is allowed for
transformative, academic, non-commercial use, except licenses that forbid even that. Redistributable assets ship with
the task; others are released as pointers (URL, version, SHA-256).

classify(raw) -> (spdx, verdict), verdict one of:
  redistributable  ships with the task (OSI licenses, CC without ND, public domain, open data licenses)
  pointer          usable, released as a pointer (arXiv's non-exclusive licence, research-only terms)
  excluded         not used: no-derivatives terms (tasks adapt assets), or no license we can verify
"""
import re

REDISTRIBUTABLE = {
    'mit', 'mit-0', 'apache-2.0', 'apache-1.1', 'bsd-2-clause', 'bsd-3-clause', 'bsd-3-clause-clear', 'bsd-4-clause',
    '0bsd', 'isc', 'zlib', 'bsl-1.0', 'mpl-2.0', 'mpl-1.1', 'epl-1.0', 'epl-2.0', 'lgpl-2.1', 'lgpl-3.0',
    'lgpl-2.1-only', 'lgpl-2.1-or-later', 'lgpl-3.0-only', 'lgpl-3.0-or-later', 'gpl-2.0', 'gpl-3.0',
    'gpl-2.0-only', 'gpl-2.0-or-later', 'gpl-3.0-only', 'gpl-3.0-or-later', 'agpl-3.0', 'agpl-3.0-only',
    'agpl-3.0-or-later', 'artistic-2.0', 'unlicense', 'wtfpl', 'cc0-1.0', 'cc-by-1.0', 'cc-by-2.0', 'cc-by-2.5',
    'cc-by-3.0', 'cc-by-4.0', 'cc-by-sa-1.0', 'cc-by-sa-2.0', 'cc-by-sa-2.5', 'cc-by-sa-3.0', 'cc-by-sa-4.0',
    'cc-by-nc-2.0', 'cc-by-nc-3.0', 'cc-by-nc-4.0', 'cc-by-nc-sa-2.0', 'cc-by-nc-sa-3.0', 'cc-by-nc-sa-4.0',
    'odbl-1.0', 'odc-by-1.0', 'pddl-1.0', 'ogl-uk-3.0', 'cdla-permissive-1.0', 'cdla-permissive-2.0',
    'cdla-sharing-1.0', 'public-domain', 'us-gov-works', 'openrail', 'openrail-m', 'creativeml-openrail-m',
    'bigscience-openrail-m', 'bigscience-bloom-rail-1.0', 'llama2', 'llama3', 'llama3.1', 'llama3.2', 'gemma',
    'afl-3.0', 'ecl-2.0', 'osl-3.0', 'cecill-2.1', 'eupl-1.2', 'ncsa', 'postgresql', 'python-2.0', 'ofl-1.1',
    'vim', 'x11', 'curl', 'libpng', 'ijg', 'cc-pdm-1.0', 'etalab-2.0', 'gfdl-1.3', 'gfdl-1.2', 'lppl-1.3c',
}
POINTER = {'arxiv-nonexclusive', 'research-only', 'other-academic', 'other-attribution', 'other-open'}
ALIASES = {
    'apache license 2.0': 'apache-2.0', 'apache 2.0': 'apache-2.0', 'apache-2': 'apache-2.0', 'apache2': 'apache-2.0',
    'mit license': 'mit', 'bsd': 'bsd-3-clause', 'bsd-new': 'bsd-3-clause', 'new bsd': 'bsd-3-clause',
    'simplified bsd': 'bsd-2-clause', 'gpl': 'gpl-3.0', 'gplv2': 'gpl-2.0', 'gplv3': 'gpl-3.0', 'lgpl': 'lgpl-3.0',
    'cc0': 'cc0-1.0', 'cc-zero': 'cc0-1.0', 'cc zero': 'cc0-1.0', 'public domain': 'public-domain', 'pd': 'public-domain',
    'publicdomain': 'public-domain', 'pd-self': 'public-domain', 'pd-old': 'public-domain', 'pd-us': 'public-domain',
    'pd-usgov': 'us-gov-works', 'public': 'public-domain', 'cc-pd': 'public-domain', 'pd-textlogo': 'public-domain',
    'odbl': 'odbl-1.0', 'odc-odbl': 'odbl-1.0', 'odc-by': 'odc-by-1.0', 'odc-pddl': 'pddl-1.0', 'pddl': 'pddl-1.0',
    'other-pd': 'public-domain', 'us-pd': 'us-gov-works', 'notspecified': '', 'noassertion': '', 'other': '',
    'unknown': '', 'none': '', 'cc-by': 'cc-by-4.0', 'cc-by-sa': 'cc-by-sa-4.0', 'cc-by-nc': 'cc-by-nc-4.0',
    'cc-by-nc-sa': 'cc-by-nc-sa-4.0', 'mpl': 'mpl-2.0', 'zlib/libpng': 'zlib', 'boost': 'bsl-1.0',
    'http://arxiv.org/licenses/nonexclusive-distrib/1.0/': 'arxiv-nonexclusive',
    'https://arxiv.org/licenses/nonexclusive-distrib/1.0/': 'arxiv-nonexclusive',
    'http://arxiv.org/licenses/assumed-1991-2003/': 'arxiv-nonexclusive',
    'https://arxiv.org/licenses/assumed-1991-2003/': 'arxiv-nonexclusive',
    'http://creativecommons.org/publicdomain/zero/1.0/': 'cc0-1.0',
    'gfdl': 'gfdl-1.3', 'attribution': 'cc-by-4.0',
    # Zenodo's generic ids (terms unknown, open or attribution-only): usable, released as pointers
    'other-at': 'other-attribution', 'other-open': 'other-open',
    # Debian DEP-5 short names
    'expat': 'mit', 'gpl-2+': 'gpl-2.0-or-later', 'gpl-3+': 'gpl-3.0-or-later', 'lgpl-2+': 'lgpl-2.1-or-later',
    'lgpl-2.1+': 'lgpl-2.1-or-later', 'lgpl-3+': 'lgpl-3.0-or-later', 'gpl-2': 'gpl-2.0', 'gpl-3': 'gpl-3.0',
    'lgpl-2.1': 'lgpl-2.1', 'agpl-3+': 'agpl-3.0-or-later', 'bsd-3': 'bsd-3-clause', 'bsd-2': 'bsd-2-clause',
    # OpenML free text
    'gpl 2': 'gpl-2.0', 'gpl-2-': 'gpl-2.0', 'database: open database, contents: database contents': 'odbl-1.0',
}
CC_URL = re.compile(r'creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)/(\d\.\d)')


def normalize(raw):
    """Free-form license text, SPDX id, CC name or URL -> lower-case id ('' when unknown)."""
    if raw is None:
        return ''
    s = str(raw).strip()
    m = CC_URL.search(s.lower())
    if m:
        kind, name, ver = m.groups()
        if kind == 'publicdomain':
            return 'cc0-1.0' if name == 'zero' else 'public-domain'
        return f'cc-{name}-{ver}'
    s = s.lower().replace('_', '-')
    if s.startswith('license:'):
        s = s[len('license:'):]
    if s in ALIASES:
        return ALIASES[s]
    # "CC BY-SA 4.0", "CC-BY-4.0", "cc by 3.0 us"
    m = re.match(r'^cc[\s\-]*(by(?:[\s\-]*(?:nc|nd|sa))*)[\s\-]*(\d\.\d)?', s)
    if m:
        parts = re.findall(r'by|nc|nd|sa', m.group(1))
        return 'cc-' + '-'.join(parts) + f'-{m.group(2) or "4.0"}'
    if re.search(r'\bcc0\b|cc-zero|public domain \(cc0\)', s):
        return 'cc0-1.0'
    m = re.search(r'\bcc[\s\-]*(by(?:[\s\-]*(?:nc|nd|sa))*)[\s\-]*(\d\.\d)?', s)   # "Attribution 4.0 International (CC BY 4.0)"
    if m:
        parts = re.findall(r'by|nc|nd|sa', m.group(1))
        return 'cc-' + '-'.join(parts) + f'-{m.group(2) or "4.0"}'
    if 'odc-by' in s or 'odc attribution' in s:
        return 'odc-by-1.0'
    if 'open database license' in s or 'odbl' in s:
        return 'odbl-1.0'
    if s.startswith('uci'):          # UCI ML repository datasets: CC BY 4.0 (site licence)
        return 'cc-by-4.0'
    s = re.sub(r'\s+', '-', s)
    return ALIASES.get(s, s)


def classify(raw):
    spdx = normalize(raw)
    if not spdx:
        return '', 'excluded'
    if '-nd' in spdx or spdx.endswith('nd'):
        return spdx, 'excluded'
    if spdx in REDISTRIBUTABLE:
        return spdx, 'redistributable'
    if spdx in POINTER:
        return spdx, 'pointer'
    return spdx, 'excluded'
