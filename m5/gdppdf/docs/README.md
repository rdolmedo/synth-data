# Shared real-document cache (M5, owner m5-gdppdf)

`/data/aai/m5/shared/docs/` holds real professional PDFs and their LiteParse text, for GDP.pdf (#357) and the
real-document lane of GDPval (#358). Built by `m5/gdppdf/docs/` on branch `m5-gdppdf`.

## Layout

| Path | What |
|---|---|
| `index.jsonl` | one line per document: `doc_id`, `source`, `texts` (shard holding its pages), `pdf_tar`, `n_pages`, `chars`, `super_tokens` (Super tokenizer, the eval's rendered page text), `too_long` (>92k: never used), `domain`, `doc_type`, `title`, `issuer`, `series`, `year`, `summary`, `richness`, `url`, `host`, `sha256`, `licence` |
| `cc2021/texts/pdfa-NNNN.jsonl.zst` | LiteParse 2.5.0 + English OCR pages per document (`pages: [{page, width, height, text}]`), exactly the GDP.pdf eval's prep config (`aa_evals/gdppdf/prepare.py`: OCR on, `eng`, tessdata_best e12c65a9, 150 DPI) |
| `cc2021/pdfs/pdfa-NNNN.tar` | the PDFs, `<doc_id>.pdf` |
| `doc_flags.jsonl` | document-level decontamination flags and near matches (`decontam/docs.py`) |

## Sources and licences

- **CC-MAIN-2021-31 PDFs** (`source: pdfa-eng-wds`): Common Crawl's July/August 2021 crawl, refetched untruncated by
  DARPA SafeDocs (digitalcorpora.org `CC-MAIN-2021-31-PDF-UNTRUNCATED`), in the English-filtered Hugging Face
  packaging `pixparse/pdfa-eng-wds` (same file ids). Each PDF keeps its publisher's copyright; Common Crawl terms of
  use apply. Used here as research training inputs only; never redistributed. Provenance per document: the SafeDocs
  file id (`id`; `cc-provenance-20230303.csv.gz` in the SafeDocs metadata maps it to the source URL) and `sha256`.
  The S3 metadata tables did not download through the proxy (stalled at <0.35 MB/s), so `url`/`host` are empty.
- Targeted sources for thin domains, if added, get their own `source` and licence line here.

## Exclusions applied before selection

- GDP.pdf test PDFs by sha256, and documents from the same issuer and series as a test PDF (`docs/exclusions.py`).
- After extraction: MinHash on the LiteParse text against the 100 test documents, whole document and per page
  (`decontam.py`); the canary GUID.
- Text above ~90k Super tokens is not selected (the eval's 131k window minus the 32k output reserve).

GDP.pdf test documents themselves are never in this cache.

## Notes

- Triage snippets of the ~648k-PDF pool were deleted after selection (the /data quota, 2026-11-12); the pool labels are
  in /data/aai/m5/gdppdf/work/pool_labels.jsonl.
- Slice 2's extraction was stopped once GDP.pdf production reached its token target; the index lists what was
  extracted (sel1 + the extracted part of sel2).
