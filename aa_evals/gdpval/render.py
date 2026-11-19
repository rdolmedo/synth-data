"""Files -> what the judge sees: text plus page images (AA: "Document-based files (.pdf, .docx, .pptx, .xlsx, etc.)
are parsed as both text and as images. We extract .zip files and parse each individual file separately.").

Runs **inside the GDPval SIF** (LibreOffice, PyMuPDF, markitdown, openpyxl, ffprobe), standard library + those:

    apptainer exec --containall -B SRC:/in:ro -B DST:/out SIF python /code/render.py /in /out

writes DST/manifest.json and DST/img/*.png. Per file: `kind`, `text` (full; the judge prompt budgets it), `images`
(rendered pages, at most MAX_PAGES per file, each <= 1 MP), `n_pages`, `notes` (what could not be shown).
- pdf: PyMuPDF text and page renders.
- docx/doc/odt/rtf/pptx/ppt/odp: markitdown text (docx/pptx) or the PDF's text, pages via LibreOffice -> PDF.
- xlsx/xls/xlsm/ods/csv: every sheet's cells as tab-separated rows (values; formulas listed after), pages via
  LibreOffice -> PDF.
- images: the image itself (<= 1 MP). psd and other raster formats via ImageMagick when possible.
- zip: extracted, each member rendered as its own file.
- audio/video: ffprobe metadata only; our open judges take neither (AA routes these to Gemini).
- text-like files: the text.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from render_cache import RenderIntegrityError, publish_png
from office_fix import OOXML, fix_ooxml   # same dir (render.py runs as a script inside the SIF)

MAX_PAGES = 20          # rendered pages per file (the prompt builder keeps fewer)
MAX_PIXELS = 1_000_000  # AA's view_image cap; also keeps judge image tokens bounded
DPI = 110
TEXT_EXT = {'.txt', '.md', '.csv', '.tsv', '.json', '.yaml', '.yml', '.xml', '.html', '.htm', '.py', '.r', '.sql',
            '.js', '.ts', '.css', '.tex', '.bib', '.log', '.ini', '.cfg', '.toml', '.sh', '.ipynb', '.overpassql',
            '.step', '.stp', '.svg', '.kml', '.geojson', '.srt', '.vtt', '.rtf_txt', '.m', '.java', '.c', '.cpp'}
OFFICE_DOC = {'.docx', '.doc', '.odt', '.rtf', '.pptx', '.ppt', '.odp', '.xlsx', '.xls', '.xlsm', '.ods', '.vsdx'}
SHEET = {'.xlsx', '.xlsm', '.xls', '.ods'}
IMAGE = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tif', '.tiff'}
OTHER_RASTER = {'.psd', '.heic', '.eps', '.ai'}
AV = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.mp4', '.mov', '.avi', '.mkv', '.webm'}
TEXT_CAP = 2_000_000    # chars kept per file in the manifest (the prompt builder truncates far below this)


def run(cmd, timeout=300, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def save_image(img_path_in, out_png):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(img_path_in)
    im.seek(0)
    im = im.convert('RGBA') if im.mode in ('P', 'LA') else im
    if im.mode == 'RGBA':
        bg = Image.new('RGB', im.size, 'white')
        bg.paste(im, mask=im.split()[3])
        im = bg
    im = im.convert('RGB')
    w, h = im.size
    if w * h > MAX_PIXELS:
        s = (MAX_PIXELS / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
    publish_png(out_png, lambda path: im.save(path, 'PNG', optimize=True))
    return list(im.size)


def pdf_pages(pdf, img_dir, stem):
    import fitz
    doc = fitz.open(pdf)
    texts, images = [], []
    for i, page in enumerate(doc):
        texts.append(page.get_text())
        if i < MAX_PAGES:
            z = DPI / 72
            # Size the render from the page geometry BEFORE rasterising (#375): a 4B deliverable had a
            # 476,889 x 200,096 pt matplotlib page, ~184 Gpx at DPI, whose first full-DPI render filled the sandbox
            # /tmp and held the collector job at 256 GB. Only pages above 64x the cap (an A0 poster is ~19x) are
            # pre-scaled, so every ordinary page takes the unchanged render-then-shrink path below.
            w, h = page.rect.width * z, page.rect.height * z
            if w * h > 64 * MAX_PIXELS:
                z *= (MAX_PIXELS / (w * h)) ** 0.5
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
            if pix.width * pix.height > MAX_PIXELS:
                z *= (MAX_PIXELS / (pix.width * pix.height)) ** 0.5
                pix = page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
            p = img_dir / f'{stem}_p{i + 1:03d}.png'
            publish_png(p, pix.save)
            images.append(p.name)
    return texts, images, len(doc)


def to_pdf(src, tmp):
    """LibreOffice -> PDF (its own profile dir per call so parallel renders don't collide)."""
    prof = Path(tempfile.mkdtemp(dir=tmp, prefix='lo_'))
    r = run(['soffice', f'-env:UserInstallation=file://{prof}', '--headless', '--convert-to', 'pdf',
             '--outdir', str(tmp), str(src)], timeout=300)
    out = Path(tmp) / (Path(src).stem + '.pdf')
    return out if out.exists() else None, (r.stderr or '')[-300:]


def markitdown_text(src):
    try:
        from markitdown import MarkItDown
        return MarkItDown().convert(str(src)).text_content
    except Exception as e:  # noqa: BLE001
        return None


def sheet_text(src):
    import openpyxl
    out, formulas = [], []
    try:
        wb_v = openpyxl.load_workbook(src, data_only=True, read_only=True)
        wb_f = openpyxl.load_workbook(src, data_only=False, read_only=True)
    except Exception as e:  # noqa: BLE001
        return None
    for ws in wb_v.worksheets:
        out.append(f'### Sheet: {ws.title}')
        for row in ws.iter_rows(values_only=True):
            if any(v is not None for v in row):
                out.append('\t'.join('' if v is None else str(v) for v in row).rstrip('\t'))
    for ws in wb_f.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str) and c.value.startswith('='):
                    formulas.append(f'{ws.title}!{c.coordinate}: {c.value}')
    if formulas:
        out.append(f'### Formulas ({len(formulas)})')
        out.extend(formulas[:2000])
    return '\n'.join(out)


def _convert_fallback(src: Path, destination: Path) -> bool:
    """Stage ImageMagick output privately; failed source decoding stays an ordinary unreadable file."""
    try:
        fd, name = tempfile.mkstemp(prefix='.' + destination.stem + '-convert-', suffix='.png',
                                    dir=destination.parent)
        os.close(fd)
    except OSError as error:
        raise RenderIntegrityError(destination, f'conversion cache staging failed: {error}') from error
    temporary = Path(name)
    try:
        result = run(['convert', f'{src}[0]', str(temporary)], timeout=120)
        try:
            save_image(temporary, destination)
        except RenderIntegrityError:
            raise
        except Exception as error:
            if result.returncode == 0:
                raise RenderIntegrityError(destination,
                                           f'convert succeeded but emitted an invalid PNG: {error}') from error
            return False
        # Some decoders report warnings/nonzero while still producing a usable image.
        return True
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # hidden staging output is never advertised in a manifest


def render_file(src: Path, rel: str, img_dir: Path, tmp: Path, idx: int) -> list:
    ext = src.suffix.lower()
    stem = f'f{idx:03d}'
    rec = {'name': rel, 'ext': ext, 'size': src.stat().st_size, 'kind': 'other', 'text': '', 'images': [],
           'n_pages': None, 'notes': []}
    try:
        if ext in OOXML:   # AA: "Office files with missing metadata patched for LibreOffice" (ours: office_fix.py)
            fixed = fix_ooxml(src.read_bytes())
            if fixed is not None:
                d = Path(tempfile.mkdtemp(dir=tmp, prefix='fix_'))
                src = d / src.name
                src.write_bytes(fixed)
                rec['notes'].append('package parts patched for LibreOffice (office_fix.py)')
        if ext == '.zip':
            d = Path(tempfile.mkdtemp(dir=tmp, prefix='zip_'))
            with zipfile.ZipFile(src) as z:
                z.extractall(d)
            recs = [dict(rec, kind='zip', notes=['extracted; members follow'])]
            members = sorted(p for p in d.rglob('*') if p.is_file() and '__MACOSX' not in p.parts)
            for j, m in enumerate(members):
                recs += render_file(m, f'{rel}/{m.relative_to(d)}', img_dir, tmp, idx * 1000 + j + 1)
            return recs
        if ext == '.pdf':
            rec['kind'] = 'pdf'
            texts, imgs, n = pdf_pages(src, img_dir, stem)
            rec.update(text='\n\n'.join(f'[page {i + 1}]\n{t}' for i, t in enumerate(texts)), images=imgs, n_pages=n)
        elif ext in OFFICE_DOC:
            rec['kind'] = 'sheet' if ext in SHEET else 'document'
            text = sheet_text(src) if ext in SHEET else None
            if text is None:
                text = markitdown_text(src)
            pdf, err = to_pdf(src, tmp)
            if pdf:
                ptexts, imgs, n = pdf_pages(pdf, img_dir, stem)
                rec.update(images=imgs, n_pages=n)
                if not text:
                    text = '\n\n'.join(ptexts)
            else:
                rec['notes'].append(f'LibreOffice could not convert the file: {err}')
            rec['text'] = text or ''
            if not text and not pdf:
                rec['notes'].append('no text or image could be extracted')
        elif ext in IMAGE or ext in OTHER_RASTER:
            rec['kind'] = 'image'
            p = img_dir / f'{stem}.png'
            try:
                save_image(src, p)
                usable = True
            except RenderIntegrityError:
                raise
            except Exception:  # noqa: BLE001
                usable = _convert_fallback(src, p)
            if usable:
                rec['images'] = [p.name]
            else:
                rec['notes'].append('image could not be decoded')
        elif ext in AV:
            rec['kind'] = 'audio/video'
            r = run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration,format_name:stream=codec_type,codec_name,width,height',
                     '-of', 'json', str(src)], timeout=60)
            rec['text'] = r.stdout[:4000]
            rec['notes'].append('audio/video content is not shown to the judge (metadata only)')
        else:
            raw = src.read_bytes()
            try:
                text = raw.decode('utf-8')
                rec['kind'] = 'text'
                if ext == '.ipynb':
                    text = notebook_text(text)
                rec['text'] = text
            except UnicodeDecodeError:
                rec['kind'] = 'binary'
                rec['notes'].append('binary file of a type the judge cannot read (name and size only)')
    except RenderIntegrityError:
        raise
    except Exception as e:  # noqa: BLE001
        rec['notes'].append(f'render failed: {type(e).__name__}: {str(e)[:300]}')
    if len(rec['text']) > TEXT_CAP:
        rec['notes'].append(f'text cut at {TEXT_CAP} chars of {len(rec["text"])}')
        rec['text'] = rec['text'][:TEXT_CAP]
    return [rec]


def notebook_text(raw):
    try:
        nb = json.loads(raw)
    except ValueError:
        return raw
    out = []
    for c in nb.get('cells', []):
        out.append(f'# [{c.get("cell_type")}]')
        out.append(''.join(c.get('source', [])))
        for o in c.get('outputs', []) or []:
            if 'text' in o:
                out.append(''.join(o['text']))
            elif 'data' in o and 'text/plain' in o['data']:
                out.append(''.join(o['data']['text/plain']))
    return '\n'.join(out)


def main(src_dir, dst_dir):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    img_dir = dst_dir / 'img'
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    tmp = Path(tempfile.mkdtemp(prefix='render_'))
    os.environ.setdefault('HOME', str(tmp))
    files = sorted(p for p in src_dir.rglob('*') if p.is_file())
    recs = []
    failure = None
    try:
        for i, f in enumerate(files):
            recs += render_file(f, str(f.relative_to(src_dir)), img_dir, tmp, i + 1)
    except RenderIntegrityError as error:
        failure = error.as_dict()
        print(str(error), file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    manifest = {'files': recs, 'n_files': len(files)}
    if failure:
        manifest.update(render_failed=True, render_integrity_error=failure)
    (dst_dir / 'manifest.json.partial').write_text(json.dumps(manifest))
    os.replace(dst_dir / 'manifest.json.partial', dst_dir / 'manifest.json')
    return failure is None


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 2)
