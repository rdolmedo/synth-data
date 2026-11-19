"""Render a ladder chat record (records.py) with the student template (Snowball), applying the cut-span loss rule:
a cut assistant message trains only the tokens of its real text, its synthetic closure is masked
(W-v2 build_data.render_record, with the real span from the text since backend-native budgets cut in Super tokens)."""
from pathlib import Path
import sys

import numpy as np

L = Path('/data/aai/m5/ladder')
sys.path[:0] = [str(L / 'worktrees/sft/sft_qwen35'), '/data/aai/m5/validation/worktrees/W-v2-code']
from data import Renderer   # noqa: E402
from tb_agents import response_budget as B   # noqa: E402

sys.path.insert(0, str(L / 'sft'))
from records import real_text   # noqa: E402

HDR = np.array([128006, 78191, 128007, 198, 14023, 771, 397])   # <|start_header_id|>assistant<|end_header_id|>\n<think>\n
EOT_ID = 128009
MAX_LEN = 131072


def renderer():
    return Renderer(str(Path(B.TOKENIZER).parent), B.TEMPLATE)


def assistant_starts(ids):
    return [int(i) + len(HDR) for i in np.flatnonzero(ids == HDR[0]) if np.array_equal(ids[i:i + len(HDR)], HDR)]


def render(R, rec):
    """Renderer.process output with the span mask applied, + audit; raises on any inconsistency."""
    msgs = [B.normalize(m) for m in rec['messages']]
    res = R.process(msgs, rec.get('tools'), rec['weights'], max_len=MAX_LEN, check_prefix_turns=0)
    assert 'error' not in res, (rec['id'], res.get('error'))
    ids, mask = res['ids'], np.array(res['mask'], dtype=np.uint8)
    starts = assistant_starts(ids)
    mi = [i for i, m in enumerate(msgs) if m['role'] == 'assistant']
    # max_len cuts at an assistant-turn end: the kept turns are the first ones
    assert len(starts) == len(mi) or (res['truncated'] and len(starts) == res['kept_turns']), (rec['id'], len(starts), len(mi))
    mi = mi[:len(starts)]
    audit = []
    for c in (c for c in rec['cut_spans'] if c in mi):
        a = starts[mi.index(c)]
        e = a + int(np.flatnonzero(ids[a:] == EOT_ID)[0]) + 1
        real = B.encode(real_text(msgs[c]))
        common = 0
        while common < min(len(real), e - a) and int(ids[a + common]) == real[common]:
            common += 1
        # a boundary retokenization may change the last few real tokens
        assert common >= len(real) - 8, (rec['id'], c, len(real), common)
        mask[a:e] = 0
        mask[a:a + common] = 1
        assert not mask[e - 1], 'synthetic EOT must not train'
        audit.append({'message_index': c, 'real_tokens': len(real), 'trained': common, 'synthetic': int(e - a - common)})
    res['mask'] = mask
    res['cut_audit'] = audit
    return res
