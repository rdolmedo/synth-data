# m5/ladder_sft: the pipeline that built the M5 ladder SFT data (mixture v2, `m5l2-*`)

Verbatim copies (2026-11-19) of the scripts that formatted and assembled the decontaminated SFT mixture `m5l2-full`
and its nested budgets `m5l2-0p33b` .. `m5l2-4p8b`, released as the Hugging Face dataset
[`ricdomolm/aai-open-sft-v0`](https://huggingface.co/datasets/ricdomolm/aai-open-sft-v0). They are kept here as
run (provenance), not refactored: paths and `sys.path` entries still name the frozen checkouts the run used.

| file | what | sha256 (= the run copy) |
|---|---|---|
| `build_suite.sh`, `build.sub` | per-source driver (CPU condor job): which record kinds run for each suite (flat single-turn suites: `t2_flat` + `fin_flat`; Pi suites: `t2_pi` + `fin_pi`; TB: `fin_pi` only), then the render pool | `f06a8417…`, `8e53f546…` |
| `records.py` | chat records `{id, group, arm, split, messages, tools, weights, cut_spans, meta}` from the trace sources (flat traces, Pi wire logs, finishing rounds); the loss rule for budget-cut turns | `61f67f63a91a5c3a82c852446ffd1abc17e4c3ebfebb062e64e8d49567a8bf71` |
| `render.py` | renders a record with the SFT chat template (`sft_qwen35/data.py` `Renderer`), applies the cut-span mask | `79f70989012e9db1a51f1f1c7178a3d9c2d2d81d871604a357ec1ddd5d187b8e` |
| `render_pool.py` | renders every record of a source into the token pool | `ae9bca4b7cee4445b414678c2f9ad089514088fd2341dbbd357e544264061a58` |
| `assemble.py` | selects the mixture at a token budget (M=2 traces per task, per-source shares), applies the decontamination exclusions, writes the packed dataset (`sft_qwen35/data.py` `write_dataset`) and its manifest | `3c7e64874e4ba447aa18607dab76d76b9bb85120d7cfecbb5139828bc161fbb9` |
| `decon_v2_exclude.json` | training-task ids excluded by the 11-17 decontamination (TB-Science-lite 1, CritPt 5, Omniscience 46) | `7162c74d3526834c94b2f2dc16f11f7d8533ad98822db6bee2ea4a97c5979bb9` |
| `assemble_all.sh`, `assemble.sub` | the assemble job | `ac0a8801…`, `1e5559a9…` |

Order: `build_suite.sh <suite>` (= `records.py <kind> <suite>`, then `render_pool.py`) for each source, then
`assemble.py --budget <B> --name <N> --exclude decon_v2_exclude.json`.

What these scripts did not generate:
- **First traces.** About 182k of the 390k train records of `m5l2-full`, and all val records, are first traces
  copied byte for byte from datasets built earlier by other code: the M5 validation datasets `W-3b` and `S-3b`, and
  the TB intermediate dataset `ib-6p0b` (`assemble.py` lines 4-6 and 45-46). These scripts render the new second
  traces and finishing rewrites and assemble everything.
- **The m5l2 job settings.** The committed `assemble.sub` still defaults to the v1 settings (`m5l-*` budgets, empty
  `EXCLUDE`). The m5l2 builds overrode its macros: one job built the four smaller budgets, and another built `full`
  alone, both with `EXCLUDE=--exclude=decon_v2_exclude.json`. Its comment says "full alone at 96" but the file
  requests 64 GB.
- `records.py` was last edited (11-15 08:45) after the flat-source records were built (05:02-07:15). That edit
  touched only the Pi code paths.

Frozen code the run imported (both commits are in this repository):
- `tb_agents/response_budget.py` from `f2467864` (byte-identical to `main` at the time of this copy). Its
  `TOKENIZER` and `TEMPLATE` constants are hard-coded paths. `TEMPLATE` is a snapshot of
  `sft_snowball/snowball_chat_template.jinja`, whose bytes match `main`.
- `sft_qwen35/data.py` (`Renderer`, `write_dataset`) from `c9fed372` (branches `m5-ladder-sft`, `m5-validation`;
  not an ancestor of `main`). `main`'s `data.py` differs by +164/−8 lines (not checked for behaviour changes);
  byte-identical rebuilds should use `c9fed372`.

To rebuild from `main`, point the `sys.path` lines at this checkout (`tb_agents/`, `sft_qwen35/`), and the
`L`/`S`/`V`/`IB` paths and response_budget's `TOKENIZER`/`TEMPLATE` at your copies.
