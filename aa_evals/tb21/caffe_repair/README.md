# Evaluator-only Caffe test-data repair (prepared, not admitted)

The historical Caffe verifier ran the submitted test network against its submitted
DataLayer source. A substituted synthetic dataset could therefore satisfy the
accuracy checks without evaluating CIFAR-10. This overlay rebinds every active
external supported LMDB Data/ImageData source in a private copy of that network to the official
10,000 test images and labels. The submitted architecture, batch sizes, transforms,
mean, resizing, grayscale/color choice, and remaining layer parameters stay intact.
A read-only first-Datum inspection requires both the canonical uint8 3×32×32
shape and exact pixel-byte membership in the already-private official TEST RGB
records. Shape alone does not prove channel order or absence of preconversion.
A distinct R/B-swapped record that also matches official data is ambiguous and
stays pending; an identical swap (R==B everywhere) is equivalent and supported.
Unrecognized, BGR, or custom-preconverted records stay typed grading-pending,
without model zero or generation retry. No BGR adapter, new dataset, uploaded
digest table, or agent-visible reference is introduced. The lookup exists only
in the trusted grader's memory after asset verification.

This is a bounded first-record domain check, not proof of whole-database identity
or every possible external preprocessing pipeline. First-Datum bytes and metadata
are archived for review; passing this check never trusts the remaining source
records or labels, which the evaluator completely replaces. Original databases
are never modified. No original raw `.bin` files are required.
It runs the same submitted checkpoint for the same 100 iterations. The >45%
threshold and <=5 percentage-point gap are unchanged. The old parser selected the
first `Batch 0, accuracy = ...` log entry despite requesting 100 iterations. The
repaired parser selects Caffe's final aggregate accuracy, as the task requires;
this is a separate scoring correction, with no retraining or generation. The old variable called
"training accuracy" is actually the **last TEST-net accuracy in the training
log**; this misleading label and comparator are deliberately preserved.

The source is the [author's CIFAR page](https://cave.cs.toronto.edu/kriz/cifar.html),
which identifies the binary archive's MD5 as `c32a1d4ab5d03f1284b67883e8d87530`.
Archive SHA256: `c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd`.
Only its unmodified `test_batch.bin` is uploaded in the verifier's tests tree;
SHA256 `8e2eb146ae340b09e24670f29cabc6326dba54da8789dab6768acf480273f65b`.
No training batches, task solution, or test asset is staged in the agent image,
setup cache, environment context, instruction, or runtime-package manifest.
Harbor uploads this tests tree inside `Verifier.verify`, after the agent phase.
The focused boundary test runs the installed Harbor phase/upload implementations
with neutral callbacks and verifies that order without any model or network call.

The public Caffe 1.0 descriptor is generated using protoc 3.12.4 from the pinned
ordinary runtime source archive. Descriptor SHA256:
`82ad194b405494013cdde22a82a6f0c4acb6f3ed4f68ea62173ed76f43b9ea3f`.
The grader bootstraps protobuf 6.33.5 and lmdb 1.7.5 only during grading. A changed
verifier script cannot match the old offline-toolchain hash, so it uses the
ordinary evaluator-network-enabled bootstrap; the agent remains offline.

Encoded-image/float/noncanonical-shape Data records, uninspectable/LevelDB sources,
HDF5Data, MemoryData, Input, Python/custom sources, legacy V1 network input syntax,
and other unsupported active external producers require an additional faithful
adapter. They raise a typed evaluator condition instead of retaining model-owned
labels or silently assigning a model failure. `/logs/verifier/grading-pending.json`
has strict schema version 1, kind `caffe_data_adapter_unsupported`, task
`caffe-cifar-10`, grading_pending true, regenerate false, a reason, and evidence
binding the submitted network, official test batch, descriptor, grader, and input
layer configurations. The script clears stale marker/reward files at start and
writes no numeric reward when a marker exists. Corrupt/missing evaluator assets
raise a distinct `GraderAssetInfrastructureError`, emit
`caffe-grader-asset-infrastructure.json`, and withhold numeric reward so the
existing missing-reward infrastructure path applies. They never emit unsupported
model-format pending evidence. **Native admission remains blocked
until the reviewed classifier recognizes this exact evidence as non-regenerating
grading pending and score-only code blocks score/CI/valid/complete.** Unknown
ordinary missing rewards retain their existing behavior.

Future artifacts capture the checkpoint/solver state and network/solver/means in
`examples/cifar10`, the training log, both supported executable/shared-lib layouts,
and Make/CMake build metadata. Dataset LMDB/LevelDB directories and static/object
libraries are excluded. Referenced means outside that directory are additionally
copied into the archived verifier logs with exact source/hash mappings. The
representation-only replay reconstructs a one-record LMDB at each original Data
source path from its captured first-Datum bytes and source/hash/shape metadata,
then calls the adapter. That single record establishes only representation;
`evaluate` still replaces it with the official test records. No synthetic or
training dataset is restored. ImageData sources are rebound directly. Replay
requires checking artifact manifest successes, the mean-file manifest, and the
pinned image/runtime-package environment; the presence of weights alone is not a
claim that the native executable and dynamic-library closure are reproducible. Dynamic dependencies outside the
build directories must be supplied by the pinned original image and package
closure; any additional agent-built external dependency needs separate capture
or a replay-infrastructure limitation, rather than an invented model result.
No extra model output files or paths are required for a task pass.

Build into a NEW out-of-git file (existing output is refused):

```bash
python -m aa_evals.tb21.caffe_repair.bundle \
  /data/aai/tbenchtrain/eval_q35/bundle/tb21_6h.tar \
  /data/aai/aa_evals/grader_assets/cifar10/v1 \
  /data/aai/aa_evals/audits/caffe_official_grader_20261107/overlay-v4-rgb-domain/tb21_caffe_official_test_v4.tar
```

The builder pins the old bundle, replaces only Caffe's accuracy function, adds
grader-only dependencies and replay capture, and independently compares every
other original member's bytes/type/mode/link. All other 88 tasks remain identical.
Task content, public schema payload, data, generated bundles, and receipts stay
outside git.

Focused neutral tests (requires isolated public protobuf/lmdb dependencies and the
out-of-git pinned assets; no Caffe checkpoint, task solution, model endpoint, or job):

```bash
PYTHONPATH=/data/aai/aa_evals/audits/caffe_official_grader_20261107/test-deps:$PWD \
  /opt/aai/venvs/harbor/bin/python \
  -m unittest aa_evals.tb21.caffe_repair.test_grader -v
```

The six historical archives represent four logical attempts; each artifact
manifest is empty. None contains checkpoint/network/training-log/executable
outputs. No corrected checkpoint regrade is possible from these archives.
A grading-only repair does not authorize replacement generation. Any future
agent-data environment decision and scoped generation are separate approvals.

The earlier native CPU proof 17873823 is **previous-domain compatibility evidence**:
its neutral Data source contained zeros, which the stricter RGB guard now leaves
pending. It demonstrated real Caffe 1.0 protobuf/LMDB, color/grayscale ImageData,
network parsing, and final accuracy output. The follow-up separately checks the
new guard against the pinned official batch and compares unchanged format,
transform, numerical payload and Caffe-call bytes with that version. No duplicate
Caffe build or old task checkpoint execution is needed or claimed.
