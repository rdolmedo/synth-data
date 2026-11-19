"""What the fit check and the generator see of an asset: its file tree, file types, heads of text files and per-format
statistics (table shape and columns, image size, media duration, SQLite tables, model tensors, PDF pages). The
generator never sees the bytes themselves, so the manifest must carry enough to write setup code against the asset.

  build(dir) -> dict      render(manifest, chars) -> text for a prompt
"""
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import zipfile
from pathlib import Path

HEAD_LINES, HEAD_LINE_CHARS, HEAD_BUDGET, TREE_MAX = 25, 240, 14000, 300
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', '.tox', '.mypy_cache'}
TEXT_EXT = {'.txt', '.md', '.rst', '.csv', '.tsv', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.xml', '.html',
            '.htm', '.tex', '.bib', '.ini', '.cfg', '.conf', '.log', '.py', '.c', '.h', '.cpp', '.hpp', '.cc', '.cu',
            '.rs', '.go', '.java', '.js', '.ts', '.rb', '.pl', '.php', '.sh', '.sql', '.r', '.jl', '.lua', '.ml',
            '.hs', '.ics', '.gcode', '.fasta', '.fa', '.gb', '.gbk', '.pdb', '.cif', '.sdf', '.mol', '.urdf', '.obo',
            '.ttl', '.rdf', '.nt', '.svg', '.srt', '.vtt', '.properties', '.service', '.lean', '.v', '.scm', '.arff'}


def walk(root):
    root = Path(root)
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files):
            p = Path(d) / f
            if p.is_symlink() or not p.is_file():
                continue
            yield p, p.relative_to(root).as_posix()


def file_types(paths):
    if not paths:
        return {}
    r = subprocess.run(['file', '-b', '-N', '-f', '-'], input='\n'.join(str(p) for p in paths), capture_output=True,
                       text=True, timeout=600)
    out = r.stdout.splitlines()
    return {str(p): (out[i][:120] if i < len(out) else '?') for i, p in enumerate(paths)}


def head_text(p):
    try:
        with open(p, 'rb') as f:
            raw = f.read(12000)
        if b'\0' in raw[:4096]:
            return None
        text = raw.decode('utf-8', errors='strict' if len(raw) < 12000 else 'ignore')
    except (OSError, UnicodeDecodeError):
        return None
    return '\n'.join(l[:HEAD_LINE_CHARS] for l in text.splitlines()[:HEAD_LINES])


def count_lines(p, cap=50_000_000):
    n = 0
    with open(p, 'rb') as f:
        while chunk := f.read(1 << 22):
            n += chunk.count(b'\n')
            if n > cap:
                break
    return n


def ffmpeg_exe():
    """A static ffmpeg (imageio-ffmpeg): the system ffprobe on some nodes fails on a libavutil/OpenCL mismatch."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which('ffmpeg') or 'ffmpeg'


def media_info(p):
    """Duration and streams from `ffmpeg -i` (stderr), e.g. 'duration 12.3s; Video: vp9 640x360, Audio: opus 48000 Hz'."""
    import re
    r = subprocess.run([ffmpeg_exe(), '-hide_banner', '-i', str(p)], capture_output=True, text=True, timeout=60)
    err = r.stderr
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', err)
    if not m:
        return '(media: could not read duration)'
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    streams = []
    for line in re.findall(r'Stream #\S+: (.*)', err)[:4]:
        kind, _, rest = line.partition(': ')
        codec = rest.split(',')[0].split(' ')[0]
        size = re.search(r', (\d{2,5}x\d{2,5})', rest)
        hz = re.search(r'(\d+) Hz, ([^,]+)', rest)
        streams.append(f'{kind}: {codec}' + (f' {size.group(1)}' if size else '') +
                       (f' {hz.group(1)} Hz {hz.group(2)}' if hz else ''))
    return f'duration {dur:.1f}s; ' + ', '.join(streams)


def stat_one(p, kind):
    """Short per-format summary; '' when nothing useful."""
    ext = p.suffix.lower()
    try:
        if ext in ('.csv', '.tsv'):
            with open(p, 'rb') as f:
                header = f.readline().decode('utf-8', 'replace').strip()[:600]
            return f'{count_lines(p)} lines; header: {header}'
        if ext == '.parquet':
            import pyarrow.parquet as pq
            m = pq.ParquetFile(p)
            cols = ', '.join(f'{f.name}:{f.type}' for f in m.schema_arrow)[:900]
            return f'{m.metadata.num_rows} rows; columns: {cols}'
        if ext == '.feather' or ext == '.arrow':
            import pyarrow.feather as pf
            t = pf.read_table(p, memory_map=True)
            return f'{t.num_rows} rows; columns: ' + ', '.join(f'{f.name}:{f.type}' for f in t.schema)[:900]
        if ext in ('.xlsx', '.xlsm'):
            import openpyxl
            wb = openpyxl.load_workbook(p, read_only=True)
            parts = []
            for ws in wb.worksheets[:8]:
                first = next(ws.iter_rows(max_row=1, values_only=True), ())
                parts.append(f'sheet {ws.title!r} {ws.max_row}x{ws.max_column}: {[c for c in first][:15]}')
            return '; '.join(parts)[:1000]
        if ext in ('.sqlite', '.sqlite3', '.db') or 'SQLite' in kind:
            con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
            tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table'")][:30]
            parts = []
            for t in tabs:
                n = con.execute(f'select count(*) from "{t}"').fetchone()[0]
                cols = [r[1] for r in con.execute(f'pragma table_info("{t}")')][:12]
                parts.append(f'{t}({", ".join(cols)}) {n} rows')
            con.close()
            return 'tables: ' + '; '.join(parts)[:1200]
        if ext in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm'):
            from PIL import Image
            with Image.open(p) as im:
                return f'{im.format} {im.width}x{im.height} {im.mode}' + (f' {getattr(im, "n_frames", 1)} frames'
                                                                         if getattr(im, 'n_frames', 1) > 1 else '')
        if ext in ('.wav', '.flac', '.ogg', '.oga', '.mp3', '.opus', '.webm', '.ogv', '.mp4', '.mkv', '.avi', '.mov'):
            return media_info(p)
        if ext == '.safetensors':
            with open(p, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                hdr = json.loads(f.read(min(n, 4 << 20)))
            ts = [(k, v.get('dtype'), v.get('shape')) for k, v in hdr.items() if k != '__metadata__']
            return f'{len(ts)} tensors; first: ' + '; '.join(f'{k} {d} {s}' for k, d, s in ts[:12])[:900]
        if ext in ('.pt', '.pth', '.bin', '.ckpt') and zipfile.is_zipfile(p):
            names = zipfile.ZipFile(p).namelist()
            return f'torch zip archive, {len(names)} entries: {names[:6]}'
        if ext == '.npy':
            with open(p, 'rb') as f:
                f.read(8)
                hl = struct.unpack('<H', f.read(2))[0]
                return 'numpy ' + f.read(hl).decode('latin1').strip()
        if ext == '.npz' and zipfile.is_zipfile(p):
            return f'npz arrays: {zipfile.ZipFile(p).namelist()[:20]}'
        if ext == '.pdf':
            r = subprocess.run(['pdfinfo', str(p)], capture_output=True, text=True, timeout=60)
            pages = next((l.split(':', 1)[1].strip() for l in r.stdout.splitlines() if l.startswith('Pages:')), '?')
            return f'PDF, {pages} pages'
        if ext in ('.json',) and p.stat().st_size < 50 << 20:
            j = json.loads(p.read_text())
            if isinstance(j, dict):
                return f'object with keys: {list(j)[:25]}'
            if isinstance(j, list):
                return f'array of {len(j)}' + (f'; first keys: {list(j[0])[:20]}' if j and isinstance(j[0], dict) else '')
        if ext == '.jsonl':
            return f'{count_lines(p)} lines'
        if ext in ('.zip', '.jar', '.docx', '.xlsx', '.pptx', '.odt', '.epub') and zipfile.is_zipfile(p):
            names = zipfile.ZipFile(p).namelist()
            return f'zip with {len(names)} entries: {names[:8]}'
    except Exception as e:  # noqa: BLE001  (best effort: a broken file still gets its type and size)
        return f'(could not inspect: {type(e).__name__})'
    return ''


BORING = ('license', 'licence', 'copying', '.gitignore', '.gitattributes', 'manifest.in', 'changelog', 'authors',
          'contributors', 'code_of_conduct', '.editorconfig', '.pre-commit', 'notice')


def boring(rel):
    name = rel.rsplit('/', 1)[-1].lower()
    return 1 if name.startswith(BORING) else 0


def build(root):
    files = list(walk(root))
    types = file_types([p for p, _ in files[:2000]])
    tree, heads, stats, budget, total = [], {}, {}, HEAD_BUDGET, 0
    for p, rel in files:
        size = p.stat().st_size
        total += size
        if len(tree) < TREE_MAX:
            tree.append([size, rel])
    # heads: the smaller text files first are often READMEs and configs; cap per file and overall
    order = sorted(files, key=lambda x: (boring(x[1]), x[1].count('/'), 0 if 'readme' in x[1].lower() else 1, x[1]))
    for p, rel in order:
        kind = types.get(str(p), '')
        if budget > 0 and (p.suffix.lower() in TEXT_EXT or 'text' in kind) and len(heads) < 40:
            h = head_text(p)
            if h:
                h = h[:min(budget, 2500)]
                heads[rel] = h
                budget -= len(h)
        if len(stats) < 60:
            s = stat_one(p, kind)
            if s:
                stats[rel] = s
    return {'n_files': len(files), 'total_bytes': total, 'tree': tree,
            'types': {rel: types.get(str(p), '?') for p, rel in files[:TREE_MAX]}, 'heads': heads, 'stats': stats}


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024


def render(m, chars=16000, paths=None):
    """Prompt text. paths: restrict to these files or directories (a selection inside the asset)."""
    def keep(rel):
        return paths is None or any(rel == q or rel.startswith(q.rstrip('/') + '/') for q in paths)
    lines = [f'{m["n_files"]} files, {human(m["total_bytes"])} in total' + (' (selection shown)' if paths else ''), '',
             '## Files (size, path, type)']
    for size, rel in m['tree']:
        if keep(rel):
            lines.append(f'{human(size):>9}  {rel}  [{m["types"].get(rel, "?")[:70]}]')
    if m['n_files'] > len(m['tree']):
        lines.append(f'... {m["n_files"] - len(m["tree"])} more files')
    st = [(rel, s) for rel, s in m['stats'].items() if keep(rel)]
    if st:
        lines += ['', '## Contents by format'] + [f'- {rel}: {s}' for rel, s in st]
    text = '\n'.join(lines)
    out = [text]
    left = chars - len(text)
    for rel, h in m['heads'].items():
        if not keep(rel):
            continue
        block = f'\n\n## Head of {rel}\n{h}'
        if len(block) > left:
            if left > 400:
                out.append(block[:left] + '\n[...]')
            break
        out.append(block)
        left -= len(block)
    return ''.join(out)
