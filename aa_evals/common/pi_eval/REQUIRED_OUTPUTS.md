# Required-output evidence (opt-in)

New Pi runs may pass `--required-output-capture-root /data/aai/aa_evals/required_outputs`.
Existing configurations remain unchanged. Only selected `train-fasttext` and
`caffe-cifar-10` trials receive selectors; other tasks use the existing path.
The root must resolve to Lustre and the namespace provider must be
`NsSidecarEnvironment`. Before agent startup, the worker writes an exclusive,
hash-bound plan under `<root>/<run_id>/<attempt_key>/` and the native attempt.
The provider validates actual task/instruction/TOML, raw-trial location, saved
run config, imported collector and frozen execution SHA. The destination must
be on a different device from raw trial storage. The initialized sandbox root
is pinned with a host directory fd before the agent runs; its native image key
is recorded, with no claim to hash the live mounted filesystem.
Binding is one-shot. Duplication and close share a lock; an unbound or closed
collector fails closed and never reopens the mutable root pathname. An I/O
thread that duplicated the pinned handle before close may finish on that handle.

The namespace provider's root is a kernel `/proc/<init-pid>/root` magic link.
Its initial pin requires the actual native `Sandbox` and its live owned asyncio
launcher handle: the launcher must be the evaluator's child and the init its
child, both owned by the evaluator as observed in host `/proc`. The init must
be PID 1 in a different PID/mount namespace. Pinned PID directory descriptors
bind the before/after PID, parent, start-tick, UID and namespace identities.
Only that authenticated kernel root entry is followed once. An arbitrary
context PID, ordinary symlink or unowned process cannot enable this path;
submitted children still use no-follow traversal. The ordinary directory API
keeps its original no-follow behavior. This fixes the pre-agent `ENOTDIR` found
by the first neutral namespace integration attempt; that failed attempt is
preserved and supplies no successful capture evidence.

Harbor collects selected files after agent execution and before tests upload.
The namespace adapter reads through that pinned root using directory-fd walks
with no-follow on every untrusted component. It executes no submitted command,
binary, `ldd`, training, or grader. It copies only regular fixed files, checks
size before reading, hashes bounded bytes, and compares fd/path identity and
size/mtime/ctime afterward. It rejects symlinks, directories, oversized and
unstable files. Nothing is copied recursively.

Files larger than 1 MiB use `/usr/bin/dd` with inherited host source/destination
fds and DIRECT writes for the 4096-byte-aligned prefix. At most 4095 tail bytes
use a buffered write to the same pinned output fd. There is no whole-file
buffered fallback if DIRECT I/O fails. Every read is bounded to the original
checked size, including the tail and the hash of the actual destination bytes.
Short reads or changed source identity/metadata never produce a full-capture
claim. Data is fsynced before publication; the containing directory and final
record file/directory are fsynced before successful capture returns. These are
filesystem durability requests, not a guarantee against storage-system failure.

FastText selects only `/app/model.bin`. Its cap is the original strict
`size < 150 * 1024 * 1024` contract (the observed valid 110,513,478-byte model
fits). Caffe selects its fixed quick TEST network, solver, iteration-500
checkpoint and training log. Capture budgets are 1 MiB per prototxt, 256 MiB
for the checkpoint, and 16 MiB for the log; these are forensic storage bounds,
**not added grading limits**. Oversized output metadata never assigns a reward.

Caffe requires the separately reviewed future overlay with exactly these fixed
artifact entries. The old broad parent-directory selectors would suppress
overlapping trial selectors, so the adapter rejects that contract before
generation. A new proposal can be prepared from our existing v4 overlay with:

```bash
python -m aa_evals.tb21.caffe_repair.bundle --narrow-artifacts OLD_V4.tar NEW_PROPOSAL.tar
```

The builder preserves every instruction, test/helper/official asset and other
task byte; only our added evaluator artifact metadata changes. Neither the
builder nor the capture flag adopts a bundle or authorizes input exposure.
Caffe replay remains explicitly incomplete: built binaries/shared libraries,
external means, custom layers and bounded input-representation evidence are
not gathered. No whole database, workspace or intermediate checkpoint sweep
is retained. Future dynamic closure needs a separate bounded design.

Independent evidence includes the plan, root binding, per-source started/final
records, captured bytes/SHA when successful, and ordinary-copy status. Missing,
unreadable, nonregular, symlink, oversized, unstable and I/O errors stay distinct.
An ordinary raw-artifact ENOSPC can coexist with a successful independent copy.
No record changes the original tests/rewards, infers a zero, repairs a file, or
provides anything to the agent. Native Harbor arguments are appended as argv
tokens; existing environment arguments are not overwritten.

Type checks and ordinary downloads have 10/30-second async bounds. A timed-out
host I/O thread can finish later; started-without-final evidence remains
incomplete, never proof of missing output. The DIRECT subprocess has its own 20-second timeout and is reaped by
the host; other blocking host filesystem syscalls are not forcibly interruptible
by the async deadline. No sandbox process or mutable root is reopened on timeout.
Snapshot stability covers the read window, not arbitrary background mutations afterward. Entire batch-worker
death can prevent collection altogether; the pre-agent plan/root binding does
not supply a missing historical model or token history. This feature addresses
controlled agent completion and later verifier-bootstrap failures, not every
external kill. Capture I/O is outside `PiCont16kEnvClock`; capture failures are
best-effort artifact failures and the original verifier still runs.
