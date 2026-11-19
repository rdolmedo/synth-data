Raw public inputs for offline tasks
==================================

The existing `--runtime-cache-manifest` and SHA256 flag pin the complete cache
release. Schema 1 keeps the original package-only behavior. Schema 2 preserves
the `tasks` package map and adds a separate `public_inputs` task map. Neither a
new default release nor a stored run is migrated automatically.

Each public input entry has `payload_root`, `instruction_sha256`,
`native_rootfs_key`, `image_probe`, `files` (relative path to SHA256),
`directories` (including empty directories), `source_index`, and `items`.
Each item has a `kind`, factual `label`, and relative `path` within that exact
inventory. Allowed kinds are `model`, `tokenizer`, `dataset`, `archive`,
`api_snapshot`, and `historical_results`. The source index is itself a pinned
payload file and records upstream URLs, revisions, retrieval dates and hashes.

Create a separate payload directory for each task. Do not point an entry at a
shared parent containing other tasks' assets. Only raw public resources required
by that task belong in its reviewed inventory. No task answer, solution code,
precomputed embedding/ranking/count, trained submission, or private verifier
asset belongs in these payloads. A public upstream archive may contain its
original public labels; retain and disclose that original source contract.

Evaluator setup validates the original instruction, native image key and package
inventory, rejects links/special files and inventory/hash changes, and uploads
the selected tree to `/opt/aa-public-inputs`. A factual notice lists local paths.
The agent still writes its requested output and installs any cached software.
Staging performs no task computation and adds no HTTP mapping, network access,
global package configuration, model preload, or grader behavior change.

Before adopting a release, verify source provenance and scoped inventories,
then run neutral canaries in each actual native image with agent networking
disabled and setup/grading networking checked separately. A repaired task
environment requires fresh generation only for that affected task; retained
results keep their original cache and source provenance.
