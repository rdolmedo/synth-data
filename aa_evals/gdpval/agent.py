"""One GDPval-AA attempt: Stirrup's Agent with AA's prompts, tools and limits, in the Apptainer sandbox.

AA (methodology, v2.1): 250 turns, a remaining-turn warning near the limit, summarization above 70% of the context,
tools code_exec / web_fetch / web_search (Brave) / view_image (vision models only) / finish / abandon_task_finish.
Stirrup's defaults already are 0.7 and a 20-turn warning; we set max_turns=250. Stirrup (unmodified) opens every run
with its base system prompt and appends AA's system prompt to it as the user's instructions; the task prompt is the
first user message.

Outcome kinds (`end`): finish (files submitted), abandon, max_turns (no finish call), output_limit (a reply hit the
output cap: Stirrup ends the run), context (the context could not be compacted below the cap), time_limit (our
wall-clock backstop; AA states none). All of these are the model's outcome and are judged as they are: a run that
submitted no files plays its matches with "no files submitted". `infra` (endpoint unreachable, sandbox failure,
harness exception) is never judged; the runner retries it.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field

from stirrup import Agent
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools import SIMPLE_FINISH_TOOL, FinishParams, ViewImageToolProvider
from stirrup.utils.logging import AgentLoggerBase

from aa_evals.gdpval import tasks as T
from aa_evals.gdpval.office_fix import fix_file
from aa_evals.gdpval.client import CappedVLLMClient, InfraFailure
from aa_evals.gdpval.sandbox import ApptainerCodeExecToolProvider
from aa_evals.gdpval.network import OFFLINE_NOTICE, require_offline

MAX_TURNS = 250


class AbandonParams(BaseModel):
    """Give up on the task: required inputs are missing, a hard dependency is unavailable, or it is incoherent."""

    reason: Annotated[str, Field(description='Brief reason the task cannot be completed.')]


async def _abandon(params: AbandonParams) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)


# AA: "Abandon Task – Signals that the model does not believe it can complete the task, with a brief reason,
# instead of submitting files." The tool name is the one AA's prompts use.
ABANDON_TOOL = Tool[AbandonParams, ToolUseCountMetadata](
    name='abandon_task_finish',
    description='Abandon the task with a brief reason instead of submitting files. Use only as a last resort, when '
                'the task genuinely cannot be completed.',
    parameters=AbandonParams,
    executor=_abandon,
)


class QuietLogger(AgentLoggerBase):
    """Counts what we report; prints nothing (hundreds of attempts share a job log)."""

    def __init__(self):
        self.name, self.model, self.max_turns, self.depth = 'gdpval', None, None, 0
        self.finish_params = self.run_metadata = self.output_dir = None
        self.counts: dict[str, int] = {}
        self.last_turn = 0

    def _inc(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def on_step(self, *a, **k):
        self._inc('steps')

    def assistant_message(self, turn, *a, **k):
        self.last_turn = max(self.last_turn, int(turn or 0))

    def user_message(self, *a, **k):
        self._inc('user_messages')

    def task_message(self, *a, **k):
        pass

    def tool_result(self, *a, **k):
        self._inc('tool_results')

    def context_summarization_start(self, *a, **k):
        self._inc('summarizations')

    def context_summarization_complete(self, *a, **k):
        self._inc('summarizations_done')

    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        self._inc('warnings')

    def error(self, *a, **k):
        self._inc('errors')


def _dump_history(history) -> list:
    """Messages as JSON, with image bytes replaced by their size (transcripts stay small)."""
    out = []
    for group in history or []:
        g = []
        for m in group:
            try:
                d = m.model_dump(mode='json')
            except Exception:  # noqa: BLE001
                d = {'repr': repr(m)[:2000]}
            g.append(_strip_images(d))
        out.append(g)
    return out


def _strip_images(x):
    if isinstance(x, dict):
        if 'data' in x and isinstance(x['data'], str) and len(x['data']) > 2000:
            return {**{k: v for k, v in x.items() if k != 'data'}, 'data': f'<{len(x["data"])} chars>'}
        return {k: _strip_images(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_images(v) for v in x]
    return x


async def run_attempt(task: T.Task, repeat: int, *, cfg: dict, work: Path, out_dir: Path) -> dict:
    """Run one attempt in `work` (node-local), write deliverables + transcript + outcome.json to `out_dir`."""
    require_offline(cfg)
    t0 = time.time()
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    home, tmp = work / 'home', work / 'tmp'
    home.mkdir(parents=True)
    tmp.mkdir()
    stats: dict[str, Any] = {}
    for src, dest in task.ref_targets():
        shutil.copy2(src, home / dest[len(T.HOME) + 1:])
        if fix_file(home / dest[len(T.HOME) + 1:]):   # AA patched these files for LibreOffice (office_fix.py)
            stats.setdefault('refs_office_patched', []).append(Path(dest).name)
    from aa_evals.gdpval.runtime_cache import prepare
    try:
        runtime_cache = await asyncio.to_thread(prepare, cfg, task, home)
    except (ValueError, OSError) as exc:
        # A setup/hash defect is infrastructure, never an ungraded zero or an infinite worker retry.
        out_dir.mkdir(parents=True, exist_ok=True)
        error = f'runtime cache setup: {type(exc).__name__}: {exc}'
        outcome = {'task_id': task.task_id, 'repeat': repeat, 'status': 'infra_error', 'end': 'infra',
                   'error': error, 'turns': 0, 'n_submitted': 0, 'submitted_files': [],
                   'wall_sec': round(time.time() - t0, 1), 'finished_at': time.time(),
                   'agent_network': cfg['agent_network'], 'sandbox': {'runtime_cache_setup_error': error}}
        (out_dir / 'outcome.json').write_text(json.dumps(outcome, indent=1))
        with gzip.open(out_dir / 'transcript.json.gz', 'wt') as f:
            json.dump({'task_id': task.task_id, 'repeat': repeat, 'history': [],
                       'events': [{'source': 'harness', 'text': error}]}, f)
        shutil.rmtree(work, ignore_errors=True)
        return outcome
    sbx_stats: dict[str, Any] = {}
    m = cfg['model']
    client = CappedVLLMClient(spec=m['spec'], served_name=m['served_name'], body=m['body'],
                              context_cap=cfg['context']['window'], max_output=m['max_output'],
                              pin_key=f'{task.task_id}/{repeat}', stats=stats, adapters=m.get('adapters', []))
    provider = ApptainerCodeExecToolProvider(sif=cfg['sandbox']['sif_local'], home=home, tmp=tmp, stats=sbx_stats,
                                                runtime_cache=runtime_cache)
    # No agent web providers; only the host-side model client can reach its endpoint.
    tools: list = [provider]
    if m['vision']:
        tools.append(ViewImageToolProvider())
    logger = QuietLogger()
    # Unmodified Stirrup always opens with its base system prompt; AA's system prompt goes in through Agent's
    # `system_prompt` (appended to the base as the user's instructions) and the task prompt is the run's message.
    agent = Agent(client=client, name='gdpval', max_turns=MAX_TURNS, tools=tools, system_prompt=T.system_prompt() + OFFLINE_NOTICE +
                          (runtime_cache['notice'] if runtime_cache else ''),
                  finish_tool=[SIMPLE_FINISH_TOOL, ABANDON_TOOL], logger=logger)
    msgs = T.task_prompt(task)
    session_out = work / 'submitted'
    finish = history = None
    end, status, error = None, 'ok', None
    try:
        async with agent.session(output_dir=str(session_out)) as session:
            finish, history, _meta = await asyncio.wait_for(session.run(msgs), timeout=cfg['limits']['wall_sec'])
        if finish is None:
            end = 'max_turns'
        elif isinstance(finish, AbandonParams):
            end = 'abandon'
        else:
            end = 'finish'
    except asyncio.TimeoutError:
        end, error = 'time_limit', f'wall-clock backstop {cfg["limits"]["wall_sec"]} s'
    except OutputTokenLimitError as e:
        end, error = 'output_limit', f'reply hit the {e.max_tokens}-token output cap'
    except ContextOverflowError as e:
        end, error = 'context', str(e)[:500]
    except InfraFailure as e:
        status, end, error = 'infra_error', 'infra', f'endpoint: {e}'
    except Exception as e:  # noqa: BLE001  a harness failure is infra, never the model's
        status, end, error = 'infra_error', 'infra', f'harness: {type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}'
    finally:
        await client.aclose()
    if status == 'ok' and sbx_stats.get('sandbox_errors', 0) and sbx_stats.get('sandbox_errors') == sbx_stats.get('commands'):
        status, end, error = 'infra_error', 'infra', f'sandbox never started: {sbx_stats.get("last_sandbox_error")}'

    out_dir.mkdir(parents=True, exist_ok=True)
    submitted = []
    if end == 'finish' and finish is not None:
        deliv = out_dir / 'deliverables'
        if deliv.exists():
            shutil.rmtree(deliv)
        if session_out.exists():
            shutil.copytree(session_out, deliv)
            submitted = sorted(str(p.relative_to(deliv)) for p in deliv.rglob('*') if p.is_file())
    home_files = sorted(str(p.relative_to(home)) for p in home.rglob('*') if p.is_file())
    # harness events for the trace judge (M3.4, generic layout): how the attempt ended and what the harness saw
    events = [{'source': 'harness', 'text': f'attempt end={end} status={status} turns={logger.last_turn} '
                                            f'submitted={len(submitted)} wall={time.time() - t0:.0f}s'},
              {'source': 'client', 'text': json.dumps(stats, default=str)[:500]},
              {'source': 'sandbox', 'text': json.dumps(sbx_stats, default=str)[:500]},
              {'source': 'agent', 'text': json.dumps(logger.counts)[:300]}]
    if finish is not None and end == 'finish':
        events.append({'source': 'harness', 'text': f'finish paths={list(getattr(finish, "paths", []) or [])[:20]} '
                                                    f'saved={submitted[:20]}'})
    with gzip.open(out_dir / 'transcript.json.gz', 'wt') as f:
        json.dump({'history': _dump_history(history), 'finish': finish.model_dump() if finish else None,
                   'events': events, 'error': error}, f)
    outcome = {
        'task_id': task.task_id, 'repeat': repeat, 'status': status, 'end': end, 'error': error,
        'finish_reason': getattr(finish, 'reason', None),
        'finish_paths': list(getattr(finish, 'paths', []) or []),
        'submitted_files': submitted, 'n_submitted': len(submitted),
        'home_files': home_files[:500], 'n_home_files': len(home_files),
        'turns': logger.last_turn, 'logger': logger.counts, 'client': stats, 'sandbox': sbx_stats,
        'agent_network': cfg['agent_network'],
        'wall_sec': round(time.time() - t0, 1), 'finished_at': time.time(),
    }
    (out_dir / 'outcome.json').write_text(json.dumps(outcome, indent=1, default=str))
    shutil.rmtree(work, ignore_errors=True)
    return outcome
