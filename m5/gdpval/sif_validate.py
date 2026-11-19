"""Runs INSIDE the GDPval SIF (python3.13 with openpyxl, python-docx, python-pptx, PyMuPDF): open every listed file in
a directory and report whether it opens, its size and a text extract (for spirit checks and decontamination).

  python3 sif_validate.py DIR NAME [NAME ...]  -> JSON on stdout: {name: {ok, error, bytes, kind, stats, text}}
"""
import json
import sys
from pathlib import Path

CAP = 20000


def xlsx(p):
    import openpyxl
    wb = openpyxl.load_workbook(p, data_only=False)
    out, cells, formulas = [], 0, 0
    for ws in wb.worksheets:
        out.append(f'## sheet {ws.title} ({ws.max_row}x{ws.max_column})')
        for row in ws.iter_rows(values_only=True):
            vals = ['' if v is None else str(v) for v in row]
            cells += sum(1 for v in vals if v)
            formulas += sum(1 for v in vals if v.startswith('='))
            if any(vals):
                out.append('\t'.join(vals).rstrip())
    return '\n'.join(out), {'sheets': len(wb.worksheets), 'cells': cells, 'formulas': formulas}


def docx(p):
    import docx as D
    d = D.Document(p)
    out = [x.text for x in d.paragraphs if x.text.strip()]
    for t in d.tables:
        for r in t.rows:
            out.append('\t'.join(c.text for c in r.cells))
    return '\n'.join(out), {'paragraphs': len(d.paragraphs), 'tables': len(d.tables)}


def pdf(p):
    import fitz
    doc = fitz.open(p)
    text = '\n'.join(pg.get_text() for pg in doc)
    return text, {'pages': doc.page_count}


def pptx(p):
    from pptx import Presentation
    pr = Presentation(p)
    out = []
    for i, s in enumerate(pr.slides):
        out.append(f'## slide {i + 1}')
        for sh in s.shapes:
            if sh.has_text_frame:
                out.append(sh.text_frame.text)
    return '\n'.join(out), {'slides': len(pr.slides)}


def text(p):
    t = Path(p).read_text(encoding='utf-8', errors='strict')
    return t, {'lines': t.count('\n') + 1}


KINDS = {'.xlsx': xlsx, '.xlsm': xlsx, '.docx': docx, '.pdf': pdf, '.pptx': pptx, '.csv': text, '.txt': text,
         '.md': text, '.json': text}


def main():
    d, names = Path(sys.argv[1]), sys.argv[2:]
    res = {}
    for n in names:
        p = d / n
        r = {'ok': False, 'bytes': p.stat().st_size if p.is_file() else None}
        fn = KINDS.get(p.suffix.lower())
        if not p.is_file():
            r['error'] = 'missing'
        elif not r['bytes']:
            r['error'] = 'empty'
        elif fn is None:
            r['error'] = 'unknown type'
        else:
            try:
                t, stats = fn(str(p))
                r.update(ok=len(t.strip()) >= 40, stats=stats, text_chars=len(t), text=t[:CAP])
                if not r['ok']:
                    r['error'] = 'no text content'
            except Exception as e:  # noqa: BLE001
                r['error'] = f'{type(e).__name__}: {str(e)[:200]}'
        res[n] = r
    print(json.dumps(res))


if __name__ == '__main__':
    main()
