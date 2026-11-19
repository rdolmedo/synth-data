"""AA: "Office files with missing metadata patched for LibreOffice". AA's patch is unpublished; this is ours.

39 GDPval Office files (38 .docx, 1 .pptx) fail in LibreOffice ("source file could not be loaded"); Word and
python-docx/pptx open them. Two package-level defects, both fixed without touching document content:
- the OPC package parts (`[Content_Types].xml`, `*.rels`) were written with ElementTree's generated prefixes
  (`<ns0:Types xmlns:ns0=...>`): rewritten with a default namespace (same elements and attributes);
- the package relationships point at metadata parts that aren't in the zip (`docProps/core.xml`, `app.xml`):
  minimal empty ones are added, with their content-type overrides.
A third defect is in the dataset itself (2 reference .docx): redacting an external link (`file:///_redacted_/...`)
dropped the `/>` that closes its `<Relationship>`, so no XML reader (Word, python-docx, LibreOffice) opens the file.
In a well-formed .rels part, `"<Relationship` can't occur, so the element is closed there.

Stdlib only: render.py imports it inside the SIF.
"""
from __future__ import annotations

import io
import re
import zipfile

OOXML = ('.docx', '.docm', '.dotx', '.xlsx', '.xlsm', '.xltx', '.pptx', '.pptm', '.potx')
_DECL = re.compile(r'\sxmlns:(ns\d+)="([^"]+)"')


def _default_ns(xml: bytes) -> bytes | None:
    """A package part that uses one generated prefix (ns0) -> the same part with a default namespace; else None."""
    try:
        s = xml.decode('utf-8')
    except UnicodeDecodeError:
        return None
    decls = _DECL.findall(s)
    if len(decls) != 1 or 'xmlns="' in s:
        return None
    p, uri = decls[0]
    s = _DECL.sub(f' xmlns="{uri}"', s, count=1)
    s = re.sub(rf'<(/?){p}:', r'<\1', s)
    if re.search(rf'\b{p}:', s):   # a prefixed attribute or something unexpected: leave the part alone
        return None
    return s.encode('utf-8')


STUBS = {
    'docProps/core.xml': ('application/vnd.openxmlformats-package.core-properties+xml',
                          '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<cp:coreProperties xmlns:cp='
                          '"http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc='
                          '"http://purl.org/dc/elements/1.1/"/>'),
    'docProps/app.xml': ('application/vnd.openxmlformats-officedocument.extended-properties+xml',
                         '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties xmlns='
                         '"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"/>'),
}


STUBS['docProps/custom.xml'] = (
    'application/vnd.openxmlformats-officedocument.custom-properties+xml',
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties xmlns='
    '"http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" xmlns:vt='
    '"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"/>')
_UNCLOSED = re.compile(r'"(\s*)(<(?:Relationship\s|/Relationships>))')


def _fix_part(name: str, xml: bytes) -> bytes | None:
    out = _default_ns(xml)
    if name.endswith('.rels'):
        s = (out or xml).decode('utf-8', 'replace')
        t = _UNCLOSED.sub(r'"/>\1\2', s)
        if t != s:
            out = t.encode('utf-8')
    return out


def _is_package_part(name: str) -> bool:
    return name == '[Content_Types].xml' or name.endswith('.rels')


def fix_ooxml(data: bytes) -> bytes | None:
    """The patched file, or None when it doesn't need (or doesn't admit) the patch."""
    try:
        zin = zipfile.ZipFile(io.BytesIO(data))
        names = zin.namelist()
    except (zipfile.BadZipFile, OSError, ValueError):
        return None
    if '[Content_Types].xml' not in names:
        return None
    fixed = {}
    for n in names:
        if _is_package_part(n):
            b = _fix_part(n, zin.read(n))
            if b is not None:
                fixed[n] = b
    rels = (fixed.get('_rels/.rels') or (zin.read('_rels/.rels') if '_rels/.rels' in names else b'')).decode(
        'utf-8', 'replace')
    targets = {t.lstrip('/') for t in re.findall(r'Target="([^"]+)"', rels)}
    add = {n: STUBS[n] for n in STUBS if n in targets and n not in names}
    if add:
        ct = (fixed.get('[Content_Types].xml') or zin.read('[Content_Types].xml')).decode('utf-8')
        for n, (ctype, _) in add.items():
            if f'PartName="/{n}"' not in ct:
                ct = re.sub(r'(</(?:\w+:)?Types>)', lambda m: f'<Override PartName="/{n}" ContentType="{ctype}"/>'
                            + m.group(1), ct, count=1)
        fixed['[Content_Types].xml'] = ct.encode('utf-8')
    if not fixed and not add:
        return None
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            zout.writestr(info, fixed.get(info.filename, zin.read(info.filename)))
        for n, (_, body) in add.items():
            zout.writestr(n, body)
    return out.getvalue()


def fix_file(path) -> bool:
    """Patch an Office file in place (a copy the caller owns). True if it changed."""
    from pathlib import Path
    p = Path(path)
    if p.suffix.lower() not in OOXML:
        return False
    b = fix_ooxml(p.read_bytes())
    if b is None:
        return False
    p.write_bytes(b)
    return True
