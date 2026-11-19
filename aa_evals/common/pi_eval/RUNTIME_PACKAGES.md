# Offline runtime package caches

New offline Pi runs can pin a reviewed public runtime-package manifest with
`--runtime-cache-manifest PATH --runtime-cache-manifest-sha256 SHA256`.
Evaluator setup verifies the manifest, original public instruction, native image
key, installed Debian inventory and every payload file before copying the cache
into `/opt/aa-runtime-cache`. Setup does not install the requested software.
The agent receives a factual cache notice and performs installation itself.

The manifest and package payloads live outside git. A versioned manifest records
per-task requirements, resolved package versions, public upstream URLs/source
commits, exact file SHA256 values, original instruction SHA256 and image identity.
The staged `public-packages.json` contains package facts and hashes. Payloads are
ordinary unmodified public packages, source archives/Git metadata and reduced
package-manager indices. They contain no benchmark verifier, reference solution,
private task data, trained weights or precomputed task output. Upstream source
releases retain their ordinary upstream files; the cache does not patch or build
the requested source project.

APT, pip and R repository settings apply only to the agent process and its tools.
The evaluator retains its original package repositories and allowed networking.
For explicitly required public Git clones, agent-local Git URL mappings point to
pinned bare source mirrors. A tool command may therefore name a public HTTPS URL
while its transport uses local files. Agent NAT and the model-only relay remain
unchanged. No general network access is enabled by the cache.

`runtime_cache_build.py` downloads package closures in the actual task image and
checks that installed Debian inventory did not change. It accepts only explicit
public package/source selectors. For example, an authorized CPU acquisition can
run `python -m aa_evals.common.pi_eval.runtime_cache_build --bundle BUNDLE --task
adaptive-rejection-sampler --packages r-base-core --out NEW_VERSION_DIRECTORY`.
The output path must be new. Acquisition requires evaluator network access;
installation proofs run in the native offline agent namespace without a model.

The reviewed public-instruction survey covers all 89 TB2.1 and 135 retained
TBScience-lite tasks. The v5 cache covers twenty TB2.1 tasks. Required named
software, build toolchains and public numerical-reference utilities qualify even
when the instruction does not use the word "install". Installation, configuration
and the task objective remain the agent's work.

| Task | Cached public requirements |
| --- | --- |
| adaptive-rejection-sampler | R runtime and APT dependency closure |
| kv-store-grpc | grpcio/grpcio-tools 1.73.0 and Python closure |
| nginx-request-logging | Nginx and APT dependency closure |
| mcmc-sampling-stan | RStan 2.32.7, CRAN dependency sources, declared system requirements |
| rstan-to-pystan | PyStan 3.10.0, Python closure and C++/Python build requirements |
| build-cython-ext | Unmodified pyknotid 0.5.3 source and declared upstream dependencies |
| build-pmars | Unmodified Debian pMARS 0.9.4 source/packaging and compiler dependencies |
| build-pov-ray | Unmodified official POV-Ray 2.2 source and compiler dependencies |
| caffe-cifar-10 | Unmodified BVLC Caffe 1.0 source and declared CPU build dependencies |
| configure-git-webserver | Git, OpenSSH client/server/SFTP server and native APT closure; no task user, repository, hook or service configuration |
| merge-diff-arc-agi-task | Git and Python for the required repository/script workflow |
| compile-compcert | Unbuilt CompCert 3.13.1 source; compatible Coq 8.16.1, OCaml 4.13.1, Menhir 20220210 and build closure |
| sqlite-with-gcov | Native C/gcov, Make and Tcl build closure; original vendored SQLite source is unchanged |
| polyglot-c-py | Python 3.12.3 runtime; existing GCC is unchanged |
| sam-cell-seg | Public MobileSAM runtime code, CPU Torch 2.7.1/torchvision 0.22.1, promised Python packages and native OpenCV dependencies; no checkpoint |
| train-fasttext | Public FastText runtime; no training data or model |
| install-windows-3.11 | QEMU runtime; no disk image or configured VM |
| make-doom-for-mips | Public native/cross compiler closure; no Doom build or task configuration |
| dna-assembly | Generic primer3/oligotm runtime; no primers or temperatures |
| dna-insert | Generic primer3/oligotm runtime; no primers or temperatures |

No retained lite task has a package-cache entry; those tasks receive no cache
notice or package-policy changes. Optional package
choices and already provided SDK packages are excluded. Models/datasets and
external API objectives remain outside this package cache: for example, Caffe
source availability does not provide CIFAR-10 data or trained weights. No task
instructions, scoring thresholds or scientific audit freezes are changed.

Run configuration and every attempt record carry the manifest path/SHA256.
Resuming a stored run with a different package-cache policy fails before workers
are submitted. Historical configs remain tied to their original execution code;
use a fresh run and explicit generation pin for a cache-enabled rerun.

The detailed instruction survey, acquisition manifests, frozen CPU source
snapshots and native offline receipts are under
`/data/aai/aa_evals/audits/pi_runtime_cache/`. Cache release and final validation
pins are recorded in that directory's `STATE.md` and the coordination status.

The default immutable release is
`/data/aai/aa_evals/runtime_packages/pi/v5/manifest.json`, SHA256
`d7fcf56be3d5c3d79d7d50e8b059772d4fcc2df11a22f39ef6f67737145fc317`.
All nine v3 entry objects and their 630 payload file hashes are unchanged. The
eleven additions have native offline installation/import/version or unrelated
synthetic compiler proofs. Cache staging leaves the required software absent;
the agent performs installation. CompCert itself is neither configured nor built,
and the compatible dependency closure runs on the original Ubuntu image without
removing or downgrading installed packages. MobileSAM is packaged from eighteen
exact public runtime/setup/license files; its upstream weight blob is not fetched.
The existing grader-only weights path is unchanged.

V5 replaces only the configure Git task payload. The other nineteen v4 entry
objects and all 1063 files remain identical. The v5 root payload admission
rehashes all 1092 files. Its neutral install proof uses the original M3.5
multiuid=false/empty-harbor-env profile and checks Git/OpenSSH versions without
creating a task user, repository, hook or HTTP service. The image's original
policy-rc.d blocks daemon autostart. Software remains uninstalled during setup;
the agent configures its own services. The separate original-task live canary
passes its saved verifier. Root source review binds its two proxy observations
to a healthy local task service and continued model transport, preserving the
model-repaired routing episode. Fresh Gemma and DeepSeek opinions are clean. The v5 review packet is under
`/data/aai/aa_evals/audits/pi_runtime_cache/v5_20261107/`.

The v4 `release-receipt.json` links all eleven stronger-profile neutral proofs
(multiuid/CPU view/memory cap enabled). These are installation checks, not exact
original M3.5 execution-profile proofs. A separate fixed fake-Pi smoke checks
that original false/empty profile, offline tool execution and model bridge. The review packet under
`/data/aai/aa_evals/audits/pi_runtime_cache/v4_20261107/` also records an exact
final-manifest production Pi smoke: two fixed fake HTTP requests, one offline
installation tool call, and successful reading of the advertised package metadata.
These are CPU runtime proofs, not benchmark scores or task-solution validation.
No real model, GPU, task objective or scientific grader is used.

The earlier v2, v3 and v4 manifests remain intact at their versioned paths.
`runtime_cache_receipt.json` retains the historical v2 survey, source snapshot and
proofs. Existing CPU tests cover integrity, empty-directory preservation,
policy pinning before submission, worker transport and oracle isolation. The
production R smoke uses the real cached Pi binary, a fixed fake OpenAI endpoint
and one install tool call; direct/proxy internet stays denied while grading can
use its own network and original package configuration. Native offline RStan
2.32.7 installation/library loading and both literal pinned upstream Git clones
also pass. Seven earlier install/import/dependency-resolution proofs apply to
unchanged payload bytes, checked against final hashes.

Failed prototypes remain in the audit directory. Initial RStan logs identified
missing CMake and BLAS/LAPACK/Fortran development libraries; the final closure
contains native `cmake`, `pandoc` and `r-base-dev`. Its neutral validation requested
32GB host memory and observed a 5,846,925,312-byte host cgroup peak with no OOM kill;
that includes cache/pagecache and does not establish a particular agent RSS bound.
Agent task limits are unchanged. Source-build tasks retain configuration/build
work for the agent; their neutral checks resolve declared build dependencies and
clone unmodified sources, without building their task objectives.
