"""The judge's view of one attempt: harness events, model turns and what the pipeline recorded.

Every run layout (layouts.py) turns its own files into a Trace. The rules (rules.py) and the LLM digest (digest.py)
read only Traces, so a new benchmark needs an adapter and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Infra-failure categories (#177). Anything else the model did is out of scope.
CATEGORIES = {
    'sandbox_kill': 'sandbox or container kill, OOM or memory-cap event that ended or crippled the attempt',
    'network': 'proxy, network or model-endpoint outage (connection refused/reset, DNS, 5xx from serving)',
    'harness_error': 'harness, chat-template or tool-call-parse error (the harness broke, not the model)',
    'harness_cutoff': 'truncation or cutoff caused by the harness or run, not by the model (run cutoff, '
                      'a cap below the protocol, an input silently dropped)',
    'silent_zero': 'an empty or missing answer caused by infra (the model never got to answer)',
    'grader_crash': 'verifier or grader crash visible in the trace (not a verifier timeout on the model\'s work)',
}


@dataclass
class Event:
    """One harness-side line: harbor.log, exception text, proxy log, verifier output, client retries..."""
    source: str
    text: str
    ref: str = ''


@dataclass
class Turn:
    """One model-side step. kind: task, text, thinking, tool_call, tool_result, error."""
    idx: int
    role: str
    kind: str
    text: str
    ref: str = ''
    meta: dict = field(default_factory=dict)


@dataclass
class Trace:
    bench: str
    key: str                      # unique per physical attempt within the run
    item_id: str
    repeat: int | None
    kind: str                     # 'agentic' or 'single'
    recorded: dict                # status, score, end_state, exception_type, infra_reason, finish_reason, ...
    events: list[Event] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    counted: bool = True          # the attempt that counts for its item (False: a retried physical attempt)
    source: str = ''              # where the trace was read from

    @property
    def known_infra(self) -> bool:
        """The pipeline already treats this attempt as infra (excluded or retried) or as a run cutoff."""
        r = self.recorded
        return r.get('status') == 'infra_error' or bool(r.get('infrastructure_failure')) or bool(r.get('cutoff'))
