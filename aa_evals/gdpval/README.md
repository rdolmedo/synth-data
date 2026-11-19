# GDPval-AA (stand-in for AA-Briefcase, 15%; issue #183)

This is AA's GDPval-AA v2.1 on our cluster. The model works agentically on each of the 220 public GDPval gold tasks and delivers files. Its deliverables are judged pairwise against the human expert's deliverable for the same task, and the results are fitted to a Bradley-Terry Elo on AA's scale. AA ranks models on GDPval-AA the way it ranks them on Briefcase (ρ = 0.97 over 137 models), which is why it holds Briefcase's slot (#173).

## Quick start

```bash
# generate on one shared model server (128k, registry settings, 5 repeats; offline agent by default)
python -m aa_evals.gdpval.submit run --run-id q38m-r1 --workers 4 --concurrency 24 \
    -- --model serving:aa-gdpval-qwen --reasoning-effort medium
python -m aa_evals.gdpval.run --run-id q38m-r1 --status
# after generation, batch judgments on one shared judge server
python -m aa_evals.gdpval.submit judge --run-id q38m-r1 --judge-arm qwen38-27b-medium-pin --judge-endpoint serving:aa-shared-judge
python -m aa_evals.gdpval.run --run-id q38m-r1 --stage score             # items.jsonl + summary.json

# several servers of the same model: a weighted pool; each attempt pins to one URL (recorded in its outcome)
python -m aa_evals.gdpval.submit run --run-id q38m-m34 ... -- --model 'serving:aa-gdpval-qwen-b200*5,serving:aa-gdpval-qwen*2' ...

# re-judge stored deliverables with another arm (judge study): no new model calls
python -m aa_evals.gdpval.submit judge --run-id q38m-r1 --judge-arm gemma4-26b-a4b-pin --judge-endpoint serving:<own server>

python -m aa_evals.gdpval.submit prepare     # once: render every task's reference files + expert deliverables
python -m aa_evals.gdpval.test_gdpval        # offline tests
```

- **Python:** a relocatable venv (Python 3.12, Stirrup 0.2.1 from the pinned clone), shipped as one file, `/data/aai/aa_evals/envs/gdpval-venv.tar` (Lustre small-file writes are slow). `job.sh` unpacks it into each condor job's scratch, and `submit` unpacks it to `/tmp/gdpval-venv` on the login node. Use `PYTHONPATH=<repo>`.
- **Sandbox image:** `/data/aai/aa_evals/sif/gdpval-v1.sif`, built from `sandbox/gdpval.def` by `sandbox/build_sif.sh` (a CPU condor job). The packages that didn't resolve are listed in `gdpval-v1.missing.txt` next to it.
- **Model settings:**
  - `--reasoning-effort` sets the effort wherever the registry's request carries one (`aa_evals/common/protocol.py`).
  - New runs freeze registry history adapters in `model.adapters`: Nemotron Super uses `inline_reasoning_history`; other profiles default to `mirror_reasoning_content`. Adapters touch only served messages, and `/tokenize` and completion receive identical adapted history. Unknown adapters fail initialization. Existing configs without this field retain their original wire behavior. Installed vLLM 0.32 also accepts legacy `reasoning_content` and restores it for templates, so this change alone is not evidence of lost thinking or a reason to regenerate historical results.
  - `--context-window` defaults to 131072. Use 0 to get the endpoint's own context; that is for Qwen3.8 calibration only.
  - `--vision on|off` overrides the image check read from the serving launch argv.

## AA's method and ours

AA's section is saved verbatim at `/data/aai/aa_evals/data/gdpval/aa_methodology_gdpval_2026-11-04.txt`.

| | AA (GDPval-AA v2.1) | Ours |
|---|---|---|
| **Data** | public gold `openai/gdpval`: 220 tasks, 44 occupations; Office files with missing metadata patched for LibreOffice | `openai/gdpval@11e7900` (parquet sha256 `f8422fab…`). The same 220 tasks, 9 sectors, 0–n reference files and the human expert's deliverables. |
| **Harness** | Stirrup, 250 turns, turn-budget warning, summarization above 70% context | **Stirrup 0.2.1 (`c092cee`) unmodified**: `max_turns=250` and Stirrup's own 0.7 cutoff and 20-turn warning |
| **Prompts** | system prompt + task prompt (published) | **verbatim task/rubric** (`aa_prompts.py`), plus a factual offline environment notice for new M3.5/M3.6 runs. Reference files are listed at `/home/user/<name>`. The "(This section appears only…)" note is removed, and the section is dropped when there are no references. Unmodified Stirrup opens every run with its own base system prompt, so AA's system prompt is passed as `Agent(system_prompt=…)`, which Stirrup appends to the base prompt. |
| **Tools** | code_exec (10-min commands), web_fetch, web_search (Brave, top 5), view_image (vision models, ≤1 MP), finish, abandon_task_finish | **No web_fetch or web_search** in new offline runs, even when a Brave key exists. Local code_exec, view_image, finish and abandon remain. |
| **Sandbox** | E2B, Debian trixie, Python 3.13, 419 pip + 762 apt packages pinned | Apptainer SIF from the same lists: apt names from the current trixie point release, pip pins exact. One `apptainer exec --containall` per command. `/home/user` and `/tmp` persist across commands; the rest of the image is read-only. Each code_exec is wrapped by an owned user/network namespace with no uplink; no agent internet. Host model clients, evaluator setup/rendering and judges retain networking. |
| **Context** | the provider's window | **131,072 tokens**, prompt plus output (M3.0). Every request is counted with vLLM `/tokenize` and gets `max_tokens` = the rest of the window. A full window triggers Stirrup's compaction. |
| **Text-only models** | no view_image | the same rule. Vision is read from the serving launch argv (`--language-model-only` / `--limit-mm-per-prompt`). |
| **Repeats** | 1 | **5** (M3 protocol) |
| **Judge** | one sampled from {GPT-5.6 Sol medium, Gemini 3.8 Flash high, Claude Opus 5 high} | **open judge**, per arm in `aa_evals/judges/arms.py`. Default `qwen38-27b-medium-pin`; the choice is the judge study on #183. |
| **Judge input** | task, reference files, submissions parsed as text + images; zips unpacked; audio/video to Gemini | the same parse (`render.py`, inside the SIF). Text: LibreOffice/markitdown/openpyxl cell dumps with formulas; images: page renders ≤1 MP. Budgeted to fit 131k: references 40k chars + 4 images; each submission 60k chars + 8 images. **Audio/video: metadata only.** |
| **Judge prompt** | not published | ours (`judge.py`, `gdpval-pairwise-v1`): which submission better responds, A / B / tie |
| **Pairs** | balanced, then Elo-active sampling across AA's model pool | **every attempt vs the human expert deliverable** of its task, judged in **both orders** (½ each). Model-vs-model matches can be added to the same fit. |
| **Fit** | Crowd-BT MLE, ties ½, sandwich CI, DeepSeek V4.1 Flash (max) = 1600 | BT MLE, ties ½, task-bootstrap CI. **Expert pinned at 968** (below). |

### Anchor: the human expert at 968

AA v2 put the human baseline at **1000** ("Human Baseline (1,000)" on the v2 leaderboard). v2.1 only re-pinned the scale, fixing DeepSeek V4.1 Flash (max) at 1600.

Wayback captures from 2026-09-15 (v2) and 2026-09-20 (v2.1) share 18 models:
- DeepSeek V4.1 Flash (max) moved from 1632.1 to 1600.0;
- the median shift was −31.4 (sd 9);
- the slope was 1.03.

So the expert sits at about **1000 − 32 = 968** on today's scale. Approved by the user via the coordinator on 2026-11-04.

**What AA's numbers imply.** If our judge matched AA's panel, Qwen3.8-27B medium (AA 1389) would beat the expert in 91.9% of judgments and xhigh (1427) in 93.4%. The other M3.1 models:

| Model | AA Elo | Win rate vs the expert |
|---|---|---|
| Super VL | 1162 | 75% |
| Qwen3.6-35B-A3B | 899 | 40% |
| Lightning | 646 | 14% |
| Gemma 4 26B-A4B | 573 | 9% |

An open judge can compress or stretch the scale. M3.7 checks our Elo against AA's over the models AA scored.

### Deviations from AA

| # | Deviation | Why | Expected effect |
|---|---|---|---|
| 1 | **Open judge** instead of a frontier panel | the user's decision (no closed API judges) | Potentially large. The judge study on #183 compares Qwen3.8-27B with the Gemma 4 judges, measures position bias, and checks how well each reproduces AA's Elo. |
| 2 | **Expert deliverable as the only opponent**, plus an anchor carried over from AA v2 | we can't reproduce AA's pool of ~300 models | The scale is AA's as of 2026-09. AA's later re-fits drift by about 10 Elo (Qwen3.8 xhigh: 1468 in v2 → 1427 now, a −41 shift against the median −31). |
| 3 | **No agent internet** (M3.5/M3.6; user, 2026-11-06) | Prevent benchmark solution access; evaluator networking is allowed | Web tools removed and kernel network isolation enforced for code_exec. Tasks requiring online research may lose capability; report separately from legacy online M3.4. |
| 3b | **Historical M3.4 only:** web_fetch through the cluster proxy (`web.py`) | nodes have no direct internet access. Stirrup's fetch ignores proxy env vars and pins the resolved IP, which the proxy refuses | None intended. Stirrup's checks, limits and extraction are unchanged, and the public-IP check still runs. Only the transport sends by hostname through the proxy. Without it, every fetch failed (smoke: 0 of 9), and models fell back to `curl` in the sandbox. Sites that block bots (Cloudflare 403) block us as they would AA. |
| 4 | **Apptainer instead of E2B**; apt versions are the current trixie point release | cluster policy | Small. The `gdpval-v1` build (sha256 `79079522…`) got all 762 apt names and 417 of 419 pip pins. `pycairo` and `pyswisseph` failed (they build from source), as listed in `gdpval-v1.missing.txt`. Background processes don't outlive their command. GPU libraries are not injected (`--no-nv`, as on E2B). |
| 5 | **Judge input budgets and no audio/video content** | 131k judge windows; no open omni judge | Long deliverables are judged on their first ~15k tokens and up to 8 page images. The 2 expert deliverables and 12 reference files that are audio or video are shown as metadata only. |
| 6 | **Judge prompt is ours** | AA's is unpublished | Unknown. |
| 6b | **System prompt composition** (Stirrup base + AA's text, as the "User" instructions) | how AA passes its system prompt is not stated; this is what unmodified Stirrup does with one | Small: both texts give the same instructions (250 steps, finish tool). |
| 7 | **5 repeats** (AA: 1) | M3 protocol | Tighter CIs, same expectation. |
| 8 | **128k context** with Stirrup compaction | M3.0 | Long tasks compact more often than on AA's native windows. Reported as `n_context_summarizations`. |
| 9 | **Office patch is ours** (AA: "files with missing metadata patched for LibreOffice", unpublished) | needed to open 39 files | `office_fix.py` repairs the package structure only, never the content. It is applied to every rendered file and to the reference files the model gets. LibreOffice failures went from 39 to 0. |
| 4b | **Sandbox memory cap of 16 GB per command** (resident set of the command's process tree; watchdog, `GDPVAL_SANDBOX_MEM_GB`) | E2B sandboxes have a fixed RAM size, set in AA's template, which is unpublished. Without a cap, one command reached 130 GB and killed a 22-attempt worker (Phase B, 2026-11-05) | A command over the cap is killed with exit 137, and the model sees "Killed: … memory limit", as with an OOM kill in a sandbox. Counted as `sandbox.memory_kills`; peak per attempt is `max_command_rss_gb`. Only the command's own process tree is watched; Stirrup and the worker are never counted or killed. |
| 10 | **Wall-clock backstop** of 6 h per attempt (AA: none) | guards against stuck runs | Counted as `n_cutoff`. |
| 11 | **Infra errors excluded** (retried up to 4×) | the shared contract (#18) | A run with more than 2% infra errors is invalid. |
| 12 | **Text-only task set** `text-only-v1` by default for new runs (`--task-set`; #346, 2026-11-11): 207 of 220 tasks, 176 of 185 anchored | our fine-tuned models are text-only and every model is served without vision | 13 tasks whose core input or deliverable is an image, audio or video are left out (`tasks.TEXT_ONLY_EXCLUDED`). Our judge never saw the audio/video ones anyway. Tasks with text inside images stay in (tesseract is in the SIF). Runs initialised earlier keep their 220-task selection; re-initialising one needs `--task-set all`. Classification: `/data/aai/aa_evals/audits/gdpval_textonly_20261111/`. |

### Dataset flaws (documented, not fixed)

- **35 of 220 tasks have no expert deliverable** (`deliverable_files` is empty in `openai/gdpval@11e7900`, and there is no text column). Every attempt on them "wins" against an empty side. They are **left out of the vs-expert fit** (185 tasks scored; `n_tasks_scored`, `n_tasks_no_expert`, and per judge `excluded_no_expert`). They still run, are judged, and count in model-vs-model pairs (#183, coordinator 2026-11-05).

These are the 576 reference and expert files rendered by `prepare` (render_v1):
- **39 Office files LibreOffice can't open** (38 .docx and 1 .pptx), the files AA patched. Word and python-docx open 37 of them. The defects:
  - 36 .docx files have their package parts written with ElementTree prefixes (`ns0:`) and no `docProps`;
  - the .pptx has relationships to `docProps/app.xml`, `core.xml` and `custom.xml`, but those parts aren't in the zip;
  - in 2 reference .docx files (tasks `01d7e53e`, `46b34f78`), redacting an external link (`file:///_redacted_/…`) removed the `/>` closing its `<Relationship>`. No XML reader can open these files, Word included.

  `office_fix.py` repairs all three cases.
- **A truncated expert deliverable:** task `0e386e32`, `PrivateCrypMixV2.zip`, is exactly 32 MiB, with no zip central directory. The judge sees a render-failure note for the expert side.
- **Unseen content:** 35 audio/video files are shown as metadata only, and 4 sheets are cut at 2M characters before the judge budget applies.

## Pi harness (#348, c30)

User decision 2026-11-11: GDPval runs under the TB 2.1 Pi harness, "so that it runs faster. performances may drop but
that is ok, as long as relative rankings are similar". The Stirrup path above stays as is (M3.5/M3.6 reproduce with it).

```bash
python -m aa_evals.gdpval.pi.tasks build                 # once: bundle (220 tasks) at /data/aai/aa_evals/data/gdpval/pi_bundle
bash aa_evals/gdpval/sandbox/build_ns_base.sh            # once (CPU job): the SIF as harbor_ns base image
python -m aa_evals.gdpval.pi.run init --run-id ID --model serving:NAME [--reasoning-effort E] [--repeats 2] \
    [--task-ids a,b] [-- PI_EVAL_ARGS]                     # pi_eval run gdpval_pi/ID + this run's config.json
python -m aa_evals.gdpval.pi.run collect --run-id ID --follow   # import + render (CPU job); then judge/score as above
```

| | Stirrup path (M3.5/M3.6) | Pi path |
|---|---|---|
| Harness | Stirrup 0.2.1, 250 turns, compaction at 70% | Pi 1.2.0 `pi-cont16k` (`pi180`), no turn cap, **no compaction**: the episode ends at the 131,072-token window |
| Output per request | the rest of the window | 16,384 tokens (continuation in Pi's history) |
| Time limit | 6 h wall backstop | 2 h of environment time (`agent_cap_sec` 7200), as TB 2.1 |
| Sandbox | `apptainer exec` of the SIF per command | harbor_ns sandbox on the same SIF's root filesystem (`docker_image = pi/tasks.py BASE_REF`), 16 GB memory cap for the sandbox, 2 CPUs, offline agent |
| Prompt | AA's system + task prompts | the same, without the finish-tool and step-budget text; deliverables = the files in `/home/user/outputs/` when the episode ends (`pi/tasks.py pi_instruction`) |
| End disclosure (accepted 2026-11-11, #348) | — (Stirrup compacts; the episode never ends at the window) | one added sentence: the session can end without warning (e.g. the window is full) and `/home/user/outputs/` is the submission as it is then (`pi/tasks.py END_DISCLOSURE`; recorded in `config.json` `harness.prompt_disclosures`) |
| Reference files | copied to `/home/user/<name>` (Office patch) | the same files and patch, staged by `NsTaskInputsEnvironment` |
| Dependency caches (3 tasks) | `runtime_cache.prepare` + per-command env | the same payload and npm cache, offline npm/pip/terraform settings as config files |
| Ends | finish / abandon / max_turns / output_limit / context / time_limit | `finish` (the agent stopped), `context` (window full), `time_limit` (env-clock cap); every end keeps its outputs and is judged |
| Infra | 4 tries | pi_eval's same-seed retries (up to 6) |

**Collection integrity.** An attempt is imported only when its trial archive, trial directory and Harbor's artifact
manifest are readable. An archive that is not visible yet (record < 1 h old) is retried on the next `collect` pass;
everything that waiting can't fix (a pack error, an archive gone for over an hour, a trial directory or artifact
manifest missing from the archive) is imported as `infra_error`, never as "no files", so `collect --follow` ends. The collected outputs must agree with the placeholder verifier, which runs in the sandbox after
collection and lists the regular files it saw (`outputs.tsv`, reward 1 iff any): a file the verifier saw but the
collection lacks, or an `empty`/`failed` collection while the verifier saw files, makes the attempt `infra_error`
(excluded, counted in `n_infra`). No files with verifier reward 0 is the model's outcome. Symbolic links in
`/home/user/outputs` are not collected (as before, only regular files are deliverables); they are listed in
`outcome.json` `links_not_collected`. If a different pi_eval attempt later resolves a key, its old render and
vs-expert matches are dropped (kept in `matches/superseded/<arm>.jsonl`) and it is rendered and judged again; this
waits while a judge stage is active on the run (it keeps `matches/judge.active.<pid>-<host>` fresh), because the
judge appends to the match file it opened. Pool (model-vs-model) matches on the old render are not invalidated:
re-plan a pool after a supersede. A background process the agent left running that still writes to the output
directory after the cap can make the verifier's list differ from the collection: that attempt is `infra_error`.

**Text only:** the port passes `--model-images off`, so pi_eval refuses a serving launch that accepts images (use
aa-serving's `*-text` specs).

Rendering, judging and both scorings (vs expert, and the two-reference ladder, #183) are unchanged: `collect` writes the
Stirrup run layout (`attempts/<task>__r<k>/{outcome.json,transcript.json.gz,deliverables/,render/}`). Seeds:
namespace `gdpval-pi-v1`.

## Model-vs-model pairs and the pooled Elo (`pool.py`)

AA's scale comes from model-vs-model pairs. Against the expert alone, the open judges saturate (#183: Qwen3.8 wins about 99.5–100%). The final GDPval-AA numbers therefore come from **one BT fit over model-vs-model pairs plus every run's vs-expert matches**, with the expert pinned at 968 (coordinator, 2026-11-05).

```bash
python -m aa_evals.gdpval.pool plan  --pool m36 --runs RUN1,RUN2,... --k 2   # once every run is complete; frozen
python -m aa_evals.gdpval.pool judge --pool m36 --judge-arm qwen38-27b-medium-pin --judge-endpoint serving:<qwen>
python -m aa_evals.gdpval.pool fit   --pool m36                                # summary.json: Elo, CI, AA's Elo, Spearman
```

- **Pairing:** each finished attempt plays `k` attempts drawn from other runs on the same task: distinct runs where possible, a random repeat, seeded per attempt. All 220 tasks count. Each pair is judged in both orders.
- **Budget:** about runs × attempts × k / 2 pairs, × 2 orders. For 6 models × 1,100 attempts at k=2, that's about 13k judge calls on top of the 2 × 925 vs-expert calls per run.
- **CIs:** bootstrap over tasks. Raise `k` if adjacent models' CIs overlap.
- **Smoke** (`_pools/dev-smoke`, 5 tasks): Gemma 4 31B preferred Qwen3.8 over Super in 8/10 judgments. Same order as AA, gap 239 vs AA's 226.

## Outcomes and scoring

- **Attempt ends:**
  - `finish`: the files are submitted.
  - `abandon`, `max_turns` (no finish within 250 turns), `output_limit` (a reply hit the output cap) and `context` (the window could not be compacted): no submission.
  - `time_limit`: also no submission.

  All of these are the model's outcome. An attempt with no submission still plays its matches, with the judge told that side submitted no files.
- **`infra`** (endpoint down, sandbox failure, harness exception) is retried and never judged.
- **summary.json:**
  - the Elo with its ci95 and the win rate vs the expert;
  - per judge: position stats (the model's score as A and as B, and the order-flip rate), ties and judge errors;
  - `ends`, `n_infra`, `n_cutoff` and `n_flagged`, plus the protocol block (model, effort, context cap, judge arm, code SHA);
  - `aa_reference`: AA's Elo and the win rate it implies.
- **items.jsonl:** one row per task × repeat (key `<task>/r<k>`): status, end, both orders' scores for the judge arm, and the transcript path.
  - The following judge job rewrites it atomically on every pass, so the **M3.4 trace judge** (`python -m aa_evals.tracejudge watch --run-dir <run> --judge serving:<tracejudge>`) reads attempts online through its generic layout.
  - Transcripts (`transcript.json.gz`) carry Stirrup's message history plus harness `events`: end state, client, sandbox and agent counters.
  - `score` folds the trace judge into summary.json (`fold_summary`: `n_flagged`) and sets `complete` once every attempt is done and judged.

## Data

- **Location:** `/data/aai/aa_evals/data/gdpval/hf_11e7900`. It is not committed: it is the public eval set, and it stays out of training data.
- **Expert and reference renders:** `/data/aai/aa_evals/data/gdpval/render_v1/<task>/{refs,expert}/`.
- **Harness source:** `/data/aai/aa_evals/data/gdpval/src/Stirrup` (`c092cee`).

### Judging completeness

Scoring keeps provisional Elo estimates for partial judge arms, with each arm labeled `partial` and counts for expected, completed and pending successful A/B judgments. `n_judgments` counts only the expert-anchored observations in the Elo fit; `n_completed_judgments` also includes the unanchored tasks. Duplicate successful retries count once. Historical judge errors remain in `judge_errors`, but never satisfy judgment completeness.

The primary arm must successfully judge both MA and MB for every successful generation before the summary is `complete` or `valid`. `generation_complete` reports generation separately. Missing generations and missing orientations are counted in `n_pending_generations` and `n_pending_judgments`; `n_pending` counts attempts still awaiting either stage. An item's final `score` remains null until both orientations succeed, while its individual `score_MA`/`score_MB` remain available. Partial secondary arms do not invalidate a completed primary arm. Existing init/work generation stages and independent judge/score stages remain unchanged.

### Judge context admission and failed-verdict retries

The legacy per-file character shares (including the2,000-character minimum) and image selection remain unchanged for every prompt that fits. Those heuristics alone do not bound aggregate context: thousands of file headers and text snippets can exceed the judge's window.

Before any chat request, `/models` supplies the actual judge context and `/tokenize` counts the full multimodal prompt with the judge's template settings. The input limit is `actual_context - output_reserve - 256`, where output reserve is the configured maximum completion length. There is no additional judge context cap. Only overfull prompts receive deterministic excerpts: rendered file text and first-page images are halved within each side, while task, rubric, section headings and final instruction stay verbatim. Exact admitted count, limit, output reserve, retained-image counts and excerpt fraction are recorded in `prompt_stats.context_budget`. If mandatory task/rubric alone cannot fit, the result remains a judge error.

This is a grading-only fix: retry failed or overfull orientations from stored deliverables, then score; no generation. Already successful judgments whose original inputs fit remain unchanged. Changed prompt metadata retains previous versions in `history`.

```bash
python -m aa_evals.gdpval.run --run-id RUN --stage judge --judge-arm qwen38-27b-medium-pin --judge-endpoint serving:shared-qwen-judge --retry-judge-errors
python -m aa_evals.gdpval.run --run-id RUN --stage score
```

Later model-vs-model pool judging uses the same token admission.

### Offline agent protocol (M3.5/M3.6)

New runs default to `--agent-network offline` and record the versioned `agent_network` policy in config, outcomes, item rows and summary. They expose no web providers, even with `BRAVE_API_KEY` present. The factual environment notice tells the model that internet and web tools are unavailable. Legacy configs/results keep their historical network label; initialization and workers reject changing a legacy network protocol, including with `--force`. Use a new run ID for the generation-wide protocol change.

Only each code_exec command is launched through `/usr/bin/unshare --user --map-root-user --net -- apptainer ...`. The owned network namespace has only loopback and no uplink, so raw sockets, direct curl, manually supplied proxy settings and calls to host model endpoints cannot restore agent egress. Namespace creation failure is a sandbox infrastructure error; it never falls back to host networking. The existing PID/proc-mount fallback may use contain+IPC on restricted hosts, but the outer network namespace remains mandatory. Writable `/home/user` and `/tmp` still persist across commands.

Stirrup's model client, evaluator preparation, reference copying, file rendering and judge requests remain on the host with networking enabled. Real CPU canaries in `test_offline.py` verify namespace separation, direct/proxy curl and IPv4/IPv6 socket failures, local file persistence, a live fake model endpoint reachable by the host client, web-provider removal and failure when namespace creation is denied. No model/GPU inference is required.

### Required dependency caches (new M3.5/M3.6 generation protocol)

The evaluator may download generic runtime/package artifacts during setup. Agents still have no egress and must install dependencies into their own attempt directories. Three tasks explicitly name implementation/test stacks absent from the pinned SIF. Their reviewed manifest contains only task IDs, instruction/template/SIF hashes, generic dependency selectors, upstream archive metadata and exact payload inventories. It contains no task text, implementation, reference answer, proof circuit, external data or credentials. Design-document technology mentions and optional tool examples do not qualify.

Evaluator setup (networking allowed; use local scratch before publishing to shared storage):

```bash
PYTHONPATH=. python aa_evals/gdpval/sandbox/build_runtime_cache.py \
  --scratch /tmp/gdpval-runtime-cache-v1 \
  --out /data/aai/aa_evals/runtime_packages/gdpval/v1 \
  --sif-sha256 79079522716486d27779ba4cef117859a999f661f702c08513dfce27fabb8d07
```

The pinned Node18.20.8/npm runtime remains an archive, for the agent to unpack into `~/.local` with existing Python tarfile/lzma support, ignoring upstream numeric ownership while preserving executable modes; nothing is installed into the SIF. The React/TypeScript test group includes React Testing Library/Sinon and generic type/runner/DOM support. The second group includes ethers and the Solidity compiler, without integration implementations or proof artifacts. The AWS group includes SDKv3, the generic Terraform binary and an AWS-provider installation mirror; no AWS request, deployment, task configuration or credential is supplied. Exact versions and transitive integrity records live in the immutable manifest. Upstream package archives are cached unmodified. Setup resolution projects, node_modules and tests remain outside the published payload.

New generation init accepts `--runtime-cache-manifest PATH` and records its digest under `sandbox.runtime_packages`. If any selected task requires this cache, the reviewed default manifest above must exist or an explicit manifest must be passed. Missing/hash-mismatched caches fail before generation. Resuming a run under a different cache protocol is rejected, including with `--force`; historical results keep their original protocol. Tasks without an explicit dependency requirement receive no additional cache, tools or package-install instructions.

Before an affected attempt starts, the worker checks the actual staged SIF hash, original task/template hashes and exact payload inventory. It copies only generic npm cache objects into an attempt-local writable cache, binds the original payload read-only at `/opt/aa-runtime-cache`, and tells the agent where the generic runtime archives and pinned selectors are. Every code_exec gets offline npm/no-index pip settings and, when relevant, a filesystem-only Terraform provider mirror. These settings are container-command arguments and attempt-local files; the host environment, model client, renderers and graders retain their networking. Setup defects produce explicit infrastructure outcomes and follow existing bounded retry/completeness rules.

CPU proof with the actual pinned SIF, neutral generic fixtures and a fake host-side model-list endpoint:

```bash
python -m aa_evals.gdpval.runtime_cache_canary \
  --manifest /data/aai/aa_evals/runtime_packages/gdpval/v1/manifest.json \
  --sif /data/aai/aa_evals/sif/gdpval-v1.sif \
  --out /data/aai/aa_evals/runtime_packages/gdpval/proofs/v1.json
```

The proof requires offline local Node/npm installation and neutral dependency tests, immutable payloads, blocked proxy/raw-socket/host-loopback access from code_exec, and an unchanged host environment with the fake model client still usable. It makes no generation request and allocates no GPU. The URL probes seen in the business canaries request external data and remain denied.

Evaluator PNG cache integrity is separate from submitted-file quality. Generated PNGs are encoded with the existing options into a temporary file, fully decoded with the SIF's Pillow, and atomically published before their page names enter the manifest. A write/decode failure publishes an explicit `render_integrity_error`; the primary vs-expert judge records `grading_pending` with no winner or numeric score and makes no model call. `--follow` does not automatically repeat that failed grading input; primary completeness and validity remain false until evaluator-only repair. Ordinary malformed submitted files keep their unreadable-file notes. Legacy empty PNGs are rejected only when the native image budget selects them, so an unused old page does not invalidate an accepted judgment. Rendering stays on CPU without requesting NVIDIA support; the unsupported Apptainer `--no-nv` spelling is omitted.

ImageMagick fallback output is staged privately as well. A successful conversion that emits an empty or corrupt PNG is a typed grading-cache failure; a failed decoder with no usable output remains an ordinary unreadable submitted file. Usable conversion output is preserved even when ImageMagick also reports a warning/nonzero status.
