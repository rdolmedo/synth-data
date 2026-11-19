# m5/gdppdf: synthetic SFT data for GDP.pdf (#357)

These are problems only: a real professional PDF plus a work request, with one Super trace each, produced with the eval's text-only protocol. There are no reference answers, rubrics or judges of traces.

## Pipeline

| Step | Code | What |
|---|---|---|
| Documents | `docs/pdfa.py`, `docs/classify.py`, `docs/select.py`, `docs/build_index.py` | See below |
| Taxonomy | `taxonomy/label_requests.py`, `taxonomy/fit.py`, `taxonomy/brainstorm.py` | See below |
| Generator | `gen/generate.py`, `gen/prompt.py`, `gen/slots.py`, `gen/checks.py` | See below |
| Traces | `trace/run.py` | See below |
| Decontamination | `decontam/docs.py`, `decontam/problems.py`, `decontam/rubric.md`, `docs/exclusions.py` | See below |
| Export | `export.py` | See below |
| Orchestration | `condor.py`, `driver.py`, `report/metrics.py` | Frozen code as git-archive tarballs, unpacked in job scratch; the production loop; the pilot and production metrics |

**Documents**
- The pool is CC-MAIN-2021-31 PDFs (SafeDocs) from the HF mirror `pixparse/pdfa-eng-wds`. 540 shards (~648k PDFs) were triaged to snippets.
- Classification into 10 domains × 11 document types ran on DeepSeek, with Super for the first 30k docs.
- Selection matches the domain and doc-type weights and GDP.pdf's page bins, with a long-document boost.
- Text is LiteParse 2.5.0 + English OCR, the eval's exact prep. The result is the shared index at `/data/aai/m5/shared/docs` (`docs/README.md`).

**Taxonomy**
- Class-level labels of the 100 test prompts and documents. Domains are weighted half on text-only-v1 and half on all 100.
- Per-domain cells are shrunk toward the global counts with α=10, and cells resting on a single test task are dropped.
- There are 2,640 leaves, Opus-brainstormed per (domain × doc type) cell. No test item was ever shown.

**Generator**
- DeepSeek V4.1 Flash, high reasoning effort, no output cap, 6 h timeout.
- It reads the whole document and writes 2–3 requests. Each slot gets a task type, up to 12 leaves from the doc's cell, asks, pitfalls, inline material, length, format, constraints and persona.
- Problem-side checks: format, single-location, answer/rubric/meta language, copying of the document, within-doc TF-IDF repeats.

**Traces**
- `render_text(prompt, pages, pdf_name)` as one text part, the `super-vl` preset (T 0.6, top_p 0.95, thinking on), a 131,072 window, a 32,768 reserve, and max_tokens = window − planned prompt.
- Problems stream in. Engines and caps come from `m5-coord/alloc.json`.

**Decontamination**
- Documents: sha256; MinHash on the whole doc (≥0.2) and on pages (≥0.5); the canary; issuer and series exclusions.
- Problems: top-30 per test task under 8-gram, TF-IDF, bge-small embedding and document-description signals, then the DeepSeek judge with 2 samples. A problem is excluded on any REUSE or on two WEAK_REUSE samples.

**Export:** `sft/prepare_data.py` format, in verified jsonl.gz shards with a MANIFEST.json of sha256 sums, cut to 330M tokens by whole documents.

## Run (as done on 2026-11-12)

```bash
python -m m5.gdppdf.condor --name X -- python -u -m <module> ...      # every heavy step is a condor job (bid 36)
python -m m5.gdppdf.docs.pdfa triage --shard N                         # x540
python -m m5.gdppdf.docs.classify --out work/pool_labels.jsonl --server m5-ds [--boost]
python -m m5.gdppdf.docs.select --labels ... --taxonomy m5/gdppdf/taxonomy/taxonomy.json --n 7200 [--have sel1] --out sel2.jsonl
python -m m5.gdppdf.docs.pdfa extract --shard N --selection sel2.jsonl
python -m m5.gdppdf.driver --selections sel1.jsonl sel2.jsonl --exclude <pilot docs> --out production
python -m m5.gdppdf.trace.run --problems 'production/problems-*.jsonl' --docs index.jsonl --out production/traces --max-tokens-total 340e6
python -m m5.gdppdf.decontam.problems retrieve|judge|decide --work decontam
python -m m5.gdppdf.export --traces ... --problems ... --excluded decontam/excluded.json --target-tokens 330e6 --out export
```

## Result (production report on #357)

`/data/aai/m5/gdppdf/export/` holds 12,666 traces, 329,996,493 tokens (Super's count of the whole sequence) and 4,588 documents.

- **Traces:** 99.98% submit an answer; 0.02% hit the context limit.
- **Decontamination:**
  - 3 problems excluded by the judge (0.023%);
  - 0 documents flagged by hash or MinHash;
  - 354 pool documents removed by the issuer and series rules.
- **Prep dry run:** `sft/prepare_data.py` keeps 122 of 122 records, with 0 template or prefix mismatches.
