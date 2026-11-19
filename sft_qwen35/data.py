"""Chat messages -> token ids + assistant loss mask -> packed SFT data for Qwen3.5.

Rendering uses the shipped Jinja chat template (sft_qwen35/chat_template.jinja: Qwen3.5's own
template plus Qwen3.8's ``preserve_thinking`` flag, default true), so every assistant turn keeps
its reasoning in later context, as the teachers saw it.

Loss spans are found by rendering prefixes:
  start of assistant turn i = len(render(msgs[:i], add_generation_prompt=True))
  end   of assistant turn i = len(render(msgs[:i+1])) - 1   (just after its <|im_end|>)
and both prefixes must be exact string prefixes of the full render (checked for every turn:
the template must be append-only for training on whole conversations to match serving).
The full string is tokenized once (add_special_tokens=False, as vLLM's chat endpoint does);
span edges must fall on token boundaries (checked), and the tokenization of each turn's prompt
prefix must equal the leading tokens of the full sequence (checked on a sample of turns).

Output layout (read by train.py's PackedDataset; same files as sft/prepare_data.py):
  tokens.npy (int32), loss_mask.npy (uint8), seq_offsets.npy (int64, len n+1),
  packs_{train,val}.json, sequences.csv, stats.json, report.md, sample_rendered.txt, COMPLETE.json (written last)

Publishing (write_dataset, #451 workstream 4): a finished `out` (COMPLETE.json) is refused before anything is built
(prepare.py / rerender.py check it before tokenizing). When `out` is on another filesystem than the node-local stage
($SFT_STAGE_DIR, else $_CONDOR_SCRATCH_DIR), the files are built and verified there (the .npy files read back against
the arrays in chunks; every record in exactly one pack, checked before stats.json is written), COMPLETE.json is added,
everything is copied with O_DIRECT (dd oflag=direct) to <out>.partial.<pid> (marker last) and renamed to <out>: it
appears complete or not at all. M5 11-15: writing tokens.npy/loss_mask.npy (27 GB) straight to /data and verifying
from there was 72 of the 112 minutes of the four nested datasets, on the SFT critical path. The job's request_disk
must hold the dataset (~5 bytes per token + ~256 per record); with too little free space write_dataset warns and
writes in place. When `out` is on the stage's filesystem (prep_job.sh builds in scratch and publishes itself; /tmp in
a condor job), or outside condor, it is written in place, verified, and COMPLETE.json is written last (atomically).
"""
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE / "chat_template.jinja"
GEN_PROMPT = "<|im_start|>assistant\n<think>\n"
IM_END = "<|im_end|>"
# Per template family: (generation prompt, end-of-turn token, text the template writes after that token).
# K2-Horizon's template writes no newline after <|ifm|im_end|> and requires a thinking field on every assistant turn.
# Snowball's (sft_snowball/snowball_chat_template.jinja) is Qwen3.5's layout on Llama-3 header tokens: no text after <|eot_id|>.
# Datakit's (open-athena Grug-67B-A2B-Datakit-SFT's own chat_template.jinja): same headers, reasoning in
# <|start_think|>...<|end_think|> inside the turn (no thinking text in the generation prompt), JSON tool calls.
MARKERS = {"qwen": (GEN_PROMPT, IM_END, "\n"),
           "k2_horizon": ("<|ifm|im_start|>assistant\n<ifm|think>\n", "<|ifm|im_end|>", ""),
           "snowball": ("<|start_header_id|>assistant<|end_header_id|>\n<think>\n", "<|eot_id|>", ""),
           "datakit": ("<|start_header_id|>assistant<|end_header_id|>\n", "<|eot_id|>", "")}
SPECIAL_STRINGS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
                   "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


class Renderer:
    """flatten_content: join text-part lists into one string ("\n"-joined), as vLLM does for templates
    with the "string" content format (e.g. Nemotron's). strip_reasoning: right-strip reasoning_content, as
    vLLM's nemotron_v3 reasoning parser does, so re-rendered history matches what the server stores."""

    def __init__(self, tokenizer_dir, template_path=DEFAULT_TEMPLATE, flatten_content=False, strip_reasoning=False):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.template = Path(template_path).read_text()
        self.template_sha = hashlib.sha256(self.template.encode()).hexdigest()
        self.flatten_content = flatten_content
        self.strip_reasoning = strip_reasoning
        self.family = ("k2_horizon" if "<|ifm|im_start|>" in self.template
                       else "datakit" if "<|start_think|>" in self.template
                       else "snowball" if "<|start_header_id|>" in self.template else "qwen")
        self.gen_prompt, self.im_end, self.turn_tail = MARKERS[self.family]

    def normalize(self, messages):
        if not (self.flatten_content or self.strip_reasoning or self.family == "k2_horizon"):
            return messages
        out = []
        for m in messages:
            m = dict(m)
            if self.family == "k2_horizon" and m["role"] == "assistant" and not isinstance(m.get("reasoning_content"), str):
                m["reasoning_content"] = ""
            if self.flatten_content and isinstance(m.get("content"), list):
                m["content"] = "\n".join(p.get("text", "") for p in m["content"]
                                         if isinstance(p, dict) and p.get("type", "text") == "text")
            if self.strip_reasoning and isinstance(m.get("reasoning_content"), str):
                m["reasoning_content"] = m["reasoning_content"].rstrip()
            out.append(m)
        return out

    def render(self, messages, tools=None, add_generation_prompt=False):
        return self.tok.apply_chat_template(messages, tools=tools, chat_template=self.template, tokenize=False,
                                            add_generation_prompt=add_generation_prompt)

    def encode_text(self, text, offsets=False):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=offsets)
        return enc

    def process(self, messages, tools=None, weights=None, max_len=None, check_prefix_turns=1, rng=None, cut_len=None,
                max_trained_turn_tokens=None):
        """Return dict(ids, mask, n_full, n_turns, kept_turns, truncated, special_in_text) or dict(error=...).

        weights[i] (default: 1 for assistant, 0 otherwise) selects which assistant turns get loss;
        zero-weight assistant turns stay in the context.
        cut_len: cut at the assistant-turn end nearest to it (never past max_len, keeping a trained turn), to give
        a sequence a target length.
        max_trained_turn_tokens: a trained turn longer than this (its tokens after the generation prompt, <|im_end|>
        included) gets zero loss: a reply the eval's per-request cap would have cut (prepare.py load_t2). Counted in
        turns_over_cap_zeroed; a record left without a trained turn is an error.
        """
        messages = self.normalize(messages)
        n = len(messages)
        if weights is None:
            weights = [1 if m["role"] == "assistant" else 0 for m in messages]
        # drop trailing non-assistant turns (nothing to learn after them)
        while n and messages[n - 1]["role"] != "assistant":
            n -= 1
        messages, weights = messages[:n], weights[:n]
        if not any(w and m["role"] == "assistant" for m, w in zip(messages, weights)):
            return {"error": "no_trained_assistant_turn"}
        try:
            full = self.render(messages, tools)
        except Exception as e:  # template raised (bad roles/order)
            return {"error": f"render:{type(e).__name__}"}
        spans = []
        for i, (m, w) in enumerate(zip(messages, weights)):
            if m["role"] != "assistant":
                continue
            pre = self.render(messages[:i], tools, add_generation_prompt=True)
            upto = self.render(messages[:i + 1], tools)
            if not (pre.endswith(self.gen_prompt) and full.startswith(pre) and full.startswith(upto)
                    and upto.endswith(self.im_end + self.turn_tail) and len(upto) > len(pre)):
                return {"error": "render_not_append_only"}
            spans.append((len(pre), len(upto) - len(self.turn_tail), bool(w), i))   # [start, end) chars, end after <|im_end|>
        # Tokenize in pieces split at every assistant-turn start (end of the generation prompt), i.e. exactly
        # as the server tokenizes the prompt and the model then emits the turn. Whole-string tokenization can
        # merge across that boundary: an empty reasoning renders "<think>\n" + "\n</think>" and the two
        # newlines become one "\n\n" token. In such cases the server's re-tokenized history differs from
        # this sequence; those turns are counted (history_retok_turns), not dropped.
        cuts = [0] + [sp[0] for sp in spans] + [len(full)]
        ids, offs = [], []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi > lo:
                enc = self.encode_text(full[lo:hi], offsets=True)
                ids += enc["input_ids"]
                offs += [(x + lo, y + lo) for x, y in enc["offset_mapping"]]
        ends = {e for _, e in offs}
        im_end_id = self.tok.convert_tokens_to_ids(self.im_end)
        mask = np.zeros(len(ids), dtype=np.uint8)
        tok_spans = []
        over_cap = 0
        for s, e, w, i in spans:
            if e not in ends:
                return {"error": "span_end_not_on_token_boundary"}
            a = _first_tok_at(offs, s)   # token indices [a, b): the turn after the generation prompt
            b = _first_tok_at(offs, e)
            assert offs[a][0] == s and offs[b - 1][1] == e and ids[b - 1] == im_end_id, "assistant span"
            if w and max_trained_turn_tokens and b - a > max_trained_turn_tokens:
                w, over_cap = False, over_cap + 1
            if w:
                mask[a:b] = 1
            tok_spans.append((a, b, w, i))
        if over_cap and not any(w for _, _, w, _ in tok_spans):
            return {"error": "no_trained_turn_within_turn_cap"}
        # The server tokenizes each turn's whole prompt string: compare with our leading tokens.
        rng = rng or random.Random(0)
        cand = list(tok_spans)
        retok = 0
        for a, b, w, i in (rng.sample(cand, min(check_prefix_turns, len(cand))) if check_prefix_turns >= 0 else cand):
            if self.encode_text(full[:offs[a][0]])["input_ids"] != ids[:a]:
                # only expected when an earlier turn had empty reasoning (see above); anything else is a bug
                if not any(m["role"] == "assistant" and not (m.get("reasoning_content") or "").strip()
                           for m in messages[:i]):
                    return {"error": "prefix_tokenization_mismatch"}
                retok += 1
        n_full = len(ids)
        truncated = False
        kept_turns = len(tok_spans)
        if cut_len and n_full > cut_len:
            first = min((b for a, b, w, i in tok_spans if w), default=None)
            turn_ends = [b for a, b, w, i in tok_spans if first is not None and first <= b and (not max_len or b <= max_len)]
            if turn_ends:
                cut = min(turn_ends, key=lambda b: (abs(b - cut_len), b))
                if cut < n_full:
                    ids, mask = ids[:cut], mask[:cut]
                    kept_turns = sum(1 for a, b, w, i in tok_spans if b <= cut)
                    truncated = True
        if max_len and len(ids) > max_len:
            fit = [b for a, b, w, i in tok_spans if b <= max_len]
            trained_fit = [b for a, b, w, i in tok_spans if b <= max_len and w]
            if not trained_fit:
                return {"error": "first_trained_turn_over_max_len"}
            cut = fit[-1]
            ids, mask = ids[:cut], mask[:cut]
            kept_turns = len(fit)
            truncated = True
        text_specials = 0
        for m in messages:
            if m["role"] in ("user", "tool", "system"):
                c = _text_of(m.get("content"))
                text_specials += sum(c.count(s) for s in SPECIAL_STRINGS)
        return {"ids": np.asarray(ids, dtype=np.int32), "mask": mask, "n_full": n_full,
                "n_turns": len(tok_spans), "kept_turns": kept_turns, "truncated": truncated,
                "special_in_text": text_specials, "prefix_checked": len(cand) if check_prefix_turns < 0 else
                min(check_prefix_turns, len(cand)), "history_retok_turns": retok,
                "empty_reasoning_turns": sum(1 for m in messages if m["role"] == "assistant"
                                             and not (m.get("reasoning_content") or "").strip()),
                **({"turns_over_cap_zeroed": over_cap} if max_trained_turn_tokens else {})}


def _first_tok_at(offs, c):
    lo, hi = 0, len(offs)
    while lo < hi:
        mid = (lo + hi) // 2
        if offs[mid][0] < c:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


# ----------------------------------------------------------------------------- packing


def ffd_pack(lengths, cap):
    """First-fit-decreasing on input lengths (n_tokens - 1). Returns list of lists of indices."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins, free = [], []
    for i in order:
        n = lengths[i]
        assert n <= cap, (n, cap)
        for b, f in enumerate(free):
            if f >= n:
                bins[b].append(i)
                free[b] -= n
                break
        else:
            bins.append([i])
            free.append(cap - n)
    return bins


def dist(a):
    a = np.asarray(a)
    if not len(a):
        return {}
    p = lambda q: int(np.percentile(a, q))
    return {"n": int(len(a)), "min": int(a.min()), "p10": p(10), "p50": p(50), "mean": int(a.mean()),
            "p90": p(90), "p99": p(99), "max": int(a.max()), "sum": int(a.sum())}


LAYOUT = ("tokens.npy", "loss_mask.npy", "seq_offsets.npy", "packs_train.json", "packs_val.json", "sequences.csv",
          "stats.json", "report.md", "sample_rendered.txt", "COMPLETE.json")


def stage_root():
    """Node-local directory to build datasets in, or None (not in a condor job and no $SFT_STAGE_DIR)."""
    d = os.environ.get("SFT_STAGE_DIR") or os.environ.get("_CONDOR_SCRATCH_DIR")
    return Path(d).resolve() if d else None


def check_out(out):
    """Refuse an `out` that holds a finished dataset (COMPLETE.json, or stats.json: datasets built before COMPLETE.json
    existed) or files that are not a dataset's: call before tokenizing, so a rerun fails at once, not after an hour of
    work. An earlier attempt that died before stats.json is replaced."""
    out = Path(out)
    if (out / "COMPLETE.json").exists() or (out / "stats.json").exists():
        raise FileExistsError(f"{out} holds a finished dataset (COMPLETE.json / stats.json): refusing to replace it; "
                              "pick a new --out or remove it")
    foreign = sorted(p.name for p in out.iterdir() if p.name not in LAYOUT and not p.name.startswith(".")) \
        if out.is_dir() else []
    if foreign:
        raise FileExistsError(f"{out} holds files that are not a dataset's ({foreign[:5]}): refusing to write there")


def _device(path):
    """st_dev of path, or of its nearest existing ancestor."""
    p = Path(path).resolve()
    while not p.exists():
        p = p.parent
    return os.stat(p).st_dev


def _save_verified(path, array, chunk=1 << 28):
    """np.save, then read the file back (memory-mapped) and compare chunk by chunk (no array-sized temporaries)."""
    np.save(path, array)
    back = np.load(path, mmap_mode="r")
    if back.dtype != array.dtype or back.shape != array.shape:
        raise OSError(f"{path}: read-back {back.dtype}{back.shape} differs from {array.dtype}{array.shape}")
    flat, bflat = array.reshape(-1), back.reshape(-1)
    for i in range(0, len(flat), chunk):
        if not np.array_equal(bflat[i:i + chunk], flat[i:i + chunk]):
            raise OSError(f"{path}: read-back differs from what was written (near element {i})")


def copy_direct(src, dst):
    """Copy one file bypassing the page cache (dd oflag=direct, size checked). Lustre writeback dirty pages count
    against the job's memory cgroup (tbshared/fastio.py; this package does not import tbshared: its frozen copies
    hold sft_qwen35/ and sft/ only). Small files, and a target without O_DIRECT (EINVAL), are copied buffered + fsync;
    any other dd failure raises."""
    size = Path(src).stat().st_size
    direct = size > 1 << 20
    if direct:
        r = subprocess.run(["dd", f"if={src}", f"of={dst}", "bs=64M", "oflag=direct", "conv=fsync", "status=none"],
                           capture_output=True, text=True)
        if r.returncode:
            if "Invalid argument" not in r.stderr:
                raise OSError(f"copy_direct: dd {src} -> {dst} failed: {r.stderr.strip()[-500:]}")
            print(f"copy_direct: no O_DIRECT on {dst} ({r.stderr.strip()[-200:]}): buffered copy", file=sys.stderr,
                  flush=True)
            direct = False
    if not direct:
        shutil.copyfile(src, dst)
        with open(dst, "rb+") as f:
            os.fsync(f.fileno())
    if Path(dst).stat().st_size != size:
        raise OSError(f"copy_direct: {dst} has {Path(dst).stat().st_size} bytes, expected {size}")


def publish(build, out):
    """build/ (complete, verified, COMPLETE.json included) -> out: every file copied to <out>.partial.<pid>, renamed
    to out, so data and marker appear together. An existing `out` must be an earlier unfinished attempt (no
    COMPLETE.json): it is replaced; a finished one is refused. A failed copy removes its .partial."""
    out = Path(out).resolve()   # a symlinked out: publish where it points
    check_out(out)
    part = out.with_name(f"{out.name}.partial.{os.getpid()}")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    try:
        for f in sorted(Path(build).iterdir(), key=lambda f: (f.name == "COMPLETE.json", f.name)):   # marker last
            copy_direct(f, part / f.name)
        if out.exists():
            shutil.rmtree(out)
        part.rename(out)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise


def write_dataset(out, records, results, pack_len, tokenizer, meta, seed=0, stage=None):
    """records[i]: dict(id, group, split, ...); results[i]: Renderer.process output (no error). stage: build dir root
    (default stage_root()); the dataset is built and verified there, then published to `out` (module docstring).
    A failed build leaves no staging directory behind."""
    try:
        return _write_dataset(out, records, results, pack_len, tokenizer, meta, seed, stage)
    except BaseException:
        root = Path(stage).resolve() if stage else stage_root()
        if root is not None:
            shutil.rmtree(root / f"sft-data-{Path(out).name}-{os.getpid()}", ignore_errors=True)
        raise


def _write_dataset(out, records, results, pack_len, tokenizer, meta, seed, stage):
    t0 = time.time()
    final = Path(out)
    check_out(final)
    root = Path(stage).resolve() if stage else stage_root()
    staged = root is not None and _device(final) != _device(root)   # out on another filesystem than the stage
    n_tokens = sum(len(r["ids"]) for r in results)
    need = 5 * n_tokens + 8 * (len(results) + 1) + 256 * len(records) + (64 << 20)
    if staged:
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        if free < need:
            print(f"write_dataset: WARNING staging in {root}: {need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free "
                  f"(raise request_disk); writing {final} in place on its own filesystem instead (slow)",
                  file=sys.stderr, flush=True)
            staged = False
    if staged:
        out = root / f"sft-data-{final.name}-{os.getpid()}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
    else:
        out = final
        out.mkdir(parents=True, exist_ok=True)
        for name in LAYOUT:   # an earlier unfinished attempt's files (e.g. a packs_val.json this build has none for)
            (out / name).unlink(missing_ok=True)
    rng = random.Random(seed)
    lens = [len(r["ids"]) for r in results]
    offsets = np.zeros(len(results) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    _save_verified(out / "tokens.npy", np.concatenate([r["ids"] for r in results]).astype(np.int32, copy=False))
    _save_verified(out / "loss_mask.npy", np.concatenate([r["mask"] for r in results]).astype(np.uint8, copy=False))
    _save_verified(out / "seq_offsets.npy", offsets)
    pack_info = {}
    for split in ("train", "val"):
        idx = [i for i, r in enumerate(records) if r["split"] == split]
        if not idx:
            continue
        in_lens = {i: lens[i] - 1 for i in idx}
        bins = ffd_pack([in_lens[i] for i in idx], pack_len)
        bins = [[idx[j] for j in b] for b in bins]
        # every record of the split in exactly one pack, none over pack_len (before stats.json marks the dir ready)
        if sorted(i for b in bins for i in b) != idx:
            raise ValueError(f"{split} packs do not hold every {split} record exactly once")
        if any(sum(in_lens[i] for i in b) > pack_len for b in bins):
            raise ValueError(f"{split}: a pack over pack_len {pack_len}")
        rng.shuffle(bins)
        for p, b in enumerate(bins):
            for i in b:
                records[i]["pack"] = f"{split}:{p}"
        fills = [sum(in_lens[i] for i in b) for b in bins]
        json.dump({"pack_len": pack_len, "pad_packs": True, "packs": bins}, open(out / f"packs_{split}.json", "w"))
        pack_info[split] = {"n_packs": len(bins), "seqs_per_pack": dist([len(b) for b in bins]),
                            "fill_efficiency": round(sum(fills) / (len(bins) * pack_len), 4),
                            "tokens": int(sum(lens[i] for i in idx)),
                            "loss_tokens": int(sum(int(results[i]["mask"].sum()) for i in idx))}
    cols = ["id", "group", "arm", "split", "pack", "n_tokens_full", "n_tokens", "n_train_tokens", "n_turns",
            "kept_turns", "truncated", "special_in_text"]
    if any("pi_bundle" in r for r in records):   # prepare.py pi_native: the Pi bundle of the record's root (last column)
        cols.append("pi_bundle")
    with open(out / "sequences.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec, res in zip(records, results):
            w.writerow({**rec, "n_tokens_full": res["n_full"], "n_tokens": len(res["ids"]),
                        "n_train_tokens": int(res["mask"].sum()), "n_turns": res["n_turns"],
                        "kept_turns": res["kept_turns"], "truncated": res["truncated"],
                        "special_in_text": res["special_in_text"]})
    # mask visualisation of the shortest sequence
    i = min(range(len(results)), key=lambda j: lens[j])
    ids, m = results[i]["ids"].tolist(), results[i]["mask"].tolist()
    parts, cur, run = [], None, []
    for t, b in zip(ids + [None], m + [None]):
        if (b != cur or t is None) and run:
            s = tokenizer.decode(run)
            parts.append(f"[[TRAIN]]{s}[[/TRAIN]]" if cur else s)
            run = []
        cur = b
        if t is not None:
            run.append(t)
    (out / "sample_rendered.txt").write_text(
        f"# id={records[i]['id']} tokens={len(ids)} trained={sum(m)}\n"
        "# [[TRAIN]]...[[/TRAIN]] = tokens with loss (reasoning + content + tool calls + <|im_end|>)\n\n" + "".join(parts))
    tr = [j for j, r in enumerate(records) if r["split"] == "train"]
    stats = {**meta, "pack_len": pack_len, "packs": pack_info,
             "tokens_train": dist([lens[j] for j in tr]),
             "tokens_full_train": dist([results[j]["n_full"] for j in tr]),
             "loss_tokens_train": dist([int(results[j]["mask"].sum()) for j in tr]),
             "truncated_train": sum(results[j]["truncated"] for j in tr),
             "special_strings_in_context_text": int(sum(r["special_in_text"] for r in results)),
             "prefix_check": {"turns_checked": int(sum(r["prefix_checked"] for r in results)),
                              "history_retok_turns": int(sum(r["history_retok_turns"] for r in results)),
                              "empty_reasoning_turns": int(sum(r["empty_reasoning_turns"] for r in results)),
                              "assistant_turns": int(sum(r["n_turns"] for r in results))}}
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    lines = [f"# SFT data: {final}", ""] + [f"- {k}: {json.dumps(v)}" for k, v in stats.items()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    built = time.time()
    done = json.dumps({"files": sorted([*(f.name for f in out.iterdir()), "COMPLETE.json"]), "tokens": int(n_tokens),
                       "records": len(records), "staged_in": str(root) if staged else None,
                       "build_s": round(built - t0, 1)}, indent=1)
    if staged:   # the marker travels with the data: out appears complete or not at all
        (out / "COMPLETE.json").write_text(done)
        publish(out, final)
        shutil.rmtree(out, ignore_errors=True)
    else:
        tmp = out / f".COMPLETE.json.{os.getpid()}"
        tmp.write_text(done)
        os.replace(tmp, out / "COMPLETE.json")
    print(f"write_dataset: {final} ({n_tokens:,} tokens) built in {built - t0:.0f} s"
          + (f", published from {root} in {time.time() - built:.0f} s" if staged else ""), file=sys.stderr, flush=True)
    return stats
