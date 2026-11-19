"""Real assets for task synthesis (issue #20): sourcing from public catalogs, a store with per-asset provenance, and
the fit check that pairs a spec with an asset.

  sources/   one adapter per catalog: search (metadata only) and fetch (download with a size cap)
  store.py   one tar.zst per asset under <ROOT>/assets/store, records in <ROOT>/assets/index/*.jsonl
  license.py SPDX normalization and the licensing policy (BLUEPRINT.md, Licensing)
  scrub.py   secrets and e-mail addresses out of text files
  manifest.py what the fit check and the generator see of an asset: tree, file types, heads, per-format stats
"""
