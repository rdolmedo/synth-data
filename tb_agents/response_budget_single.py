"""Budgeted single-turn tasks become interrupted assistant + notice + finish.

Keep the benchmark's final-answer extraction and grader unchanged. Store the full
interrupted conversation and count only actual retained model body tokens.
"""
import copy
from tb_agents import response_budget as B
from tb_agents.pi_context_limit import with_notice


def request_payload(payload):
    """vLLM 0.20.1 chat parsing reads `reasoning` from request histories.

    Keep canonical rc in stored trajectories, and send its current API alias too.
    The server exposes both names to the checkpoint template after parsing.
    """
    out = copy.deepcopy(payload)
    for message in out.get('messages', []):
        if message.get('role') == 'assistant' and message.get('reasoning_content'):
            message['reasoning'] = message['reasoning_content']
    return out


def step(payload, result, used, budget, messages):
    cap = B.output_cap(used,budget)
    c = result['choices'][0]
    if budget.get('token_counting') == 'backend-native':
        generated = (result.get('usage') or {}).get('completion_tokens')
        if type(generated) is not int or not 0 <= generated <= cap:
            raise ValueError('Native response budget requires valid backend completion usage within the requested cap')
        msg = B.normalize(c['message'])
        ids = B.encode(B.body(msg))
        clipped = c['finish_reason'] == 'length'
        used += generated
    elif budget.get('token_counting') == 'super':   # one unit for every model: Super tokens of the body
        msg, ids, clipped, n = B.super_clip(c['message'], cap)
        used += n
    else:
        msg, ids, clipped = B.clip(c['message'], cap)
        used += len(ids)
    done = not clipped and c['finish_reason'] not in ('length', None)
    messages.append(msg)
    return msg, ids, clipped, used, done


TURN_SLACK = 32   # per call, when the backend reports no prompt_tokens: template tokens of a re-sent turn + CONTINUE


class Window:
    """The near-window bound in the MODEL's tokens. item_budget fits the warning and the total to the item's context
    room (`max_output` = window - prompt bound, as the runners compute max_tokens) in the budget's units; with
    token_counting 'super' those are Super tokens of the body, while every continuation re-sends the body in the
    model's own tokens. For a model whose tokens run ahead of Super's on its own output, the prompt outgrew the window
    and vLLM rejected the call (HTTP 400 'maximum context length', recorded as an infra error and regenerated on
    resume: m5l-18k-1p0b HLE, 2026-11-15, 34 of its first 50 items, each after ~100k generated tokens).

    This tracks the prompt's real growth from the backend's usage (the last call's prompt + completion tokens over
    the first call's prompt; an upper bound when a body was clipped) and keeps the near-window rule in those units:
    before the warning a call stops where `allowance` tokens of room are left, the warning comes there at the latest,
    and after it each call is capped to the room left (stated in the notice when smaller). The first call is unchanged;
    while the model's tokens grow like the budget's (Super) it moves the warning by at most the few template tokens of
    the re-sent turns, which NEAR_WINDOW_MARGIN already reserved. Off without near_window or without backend usage."""

    def __init__(self, budget, max_output):
        self.on = bool(budget and budget.get('near_window') and max_output)
        self.room = int(max_output) - B.NEAR_WINDOW_MARGIN if self.on else 0
        self.allow = budget['finish_response_tokens'] if self.on else 0
        self.p0, self.grown, self.n, self.warned, self.forced = None, 0, 0, False, False

    def update(self, usage):
        u = usage or {}
        pt, ct = u.get('prompt_tokens'), u.get('completion_tokens')
        if self.n == 0 and type(pt) is int:
            self.p0 = pt
        if self.p0 is not None and type(pt) is int and type(ct) is int:
            self.grown = max(self.grown, pt + ct - self.p0)
        elif type(ct) is int:
            self.grown += ct + TURN_SLACK
        self.n += 1

    def left(self):
        return self.room - self.grown

    def cap(self, used, budget):
        """max_tokens for the next call (before the `requested` bound); < 1: the room is spent, stop."""
        cap = B.output_cap(used, budget)
        if not self.on:
            return cap
        if not self.warned:
            return min(cap, self.left() - self.allow)
        if budget.get('measure_backstop_tokens'):
            return min(cap, self.left())
        left, r = self.stated(used, budget), B.reserve(budget)
        return left - r if left > r else left

    def warn(self, used, budget):
        """Whether the cut is followed by CUT + notice (sticky once given)."""
        if used >= budget['warn_after_response_tokens']:
            self.warned = True
        elif self.on and not self.warned and self.left() <= self.allow:
            self.warned = self.forced = True
        return self.warned

    def stated(self, used, budget):
        left = budget['total_response_tokens'] - used
        return min(left, self.left()) if self.on else left

    def record(self):
        return {'room': self.room, 'grown': self.grown, 'forced_warning': self.forced} if self.on else None


def run(call, payload, budget):
    payload = copy.deepcopy(payload)
    budget = B.item_budget(budget, payload.get('max_tokens'))
    requested=int(payload.get('max_tokens')or budget['total_response_tokens'])
    g = Window(budget, payload.get('max_tokens'))
    messages = payload['messages']
    used, calls, totals = 0, [], {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    finish = 'length'
    while used < budget['total_response_tokens'] and len(calls) < 16:
        cap = g.cap(used, budget)
        if cap < 1 and calls:   # the model's context room is spent: a length stop
            break
        payload['max_tokens'] = max(1, min(requested, cap))
        payload['messages'] = messages
        result = call(request_payload(payload))
        msg, ids, clipped, used, done = step(payload, result, used, budget, messages)
        g.update(result.get('usage'))
        calls.append({'response': result, 'real_body_ids': ids, 'clipped': clipped,
                      'message_index': len(messages)-1})
        for k in totals:
            totals[k] += (result.get('usage') or {}).get(k) or 0
        if done:
            finish = result['choices'][0]['finish_reason']
            break
        if used >= budget['total_response_tokens']:
            break
        warned = g.warn(used, budget)
        messages.append({'role': 'user', 'content': B.CUT if warned else B.CONTINUE})
        if warned:
            left = g.stated(used, budget)
            messages = with_notice(messages, B.notice(left,final=left<=B.reserve(budget),extra=budget.get('notice_extra')))
    result = copy.deepcopy(calls[-1]['response'])
    reasoning = '\n\n'.join(m.get('reasoning_content') or '' for m in messages if m['role'] == 'assistant')
    content = '\n\n'.join(m.get('content') or '' for m in messages if m['role'] == 'assistant')
    result['choices'][0].update(message={'role': 'assistant', 'reasoning_content': reasoning, 'content': content}, finish_reason=finish)
    result['usage'] = totals
    result['_response_budget'] = {'policy': B.POLICY, 'budget': budget, 'used': used,
                                  'messages': messages, 'calls': calls, 'window': g.record()}
    return result


async def run_async(call, payload, budget):
    """Adapter for GDP.pdf's existing asynchronous client response shape."""
    payload = copy.deepcopy(payload)
    budget = B.item_budget(budget, payload.get('max_tokens'))
    requested=int(payload.get('max_tokens')or budget['total_response_tokens'])
    g = Window(budget, payload.get('max_tokens'))
    messages = payload['messages']
    used, calls, totals, finish = 0, [], {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, 'length'
    while used < budget['total_response_tokens'] and len(calls) < 16:
        cap = g.cap(used, budget)
        if cap < 1 and calls:   # the model's context room is spent: a length stop
            break
        payload['max_tokens'] = max(1, min(requested, cap))
        payload['messages'] = messages
        out = await call(request_payload(payload))
        result = {'choices': [{'message': {'role': 'assistant', 'content': out['content'],
                   'reasoning_content': out['reasoning']}, 'finish_reason': out['finish_reason']}],
                  'usage': out.get('usage') or {}}
        msg, ids, clipped, used, done = step(payload, result, used, budget, messages)
        g.update(out.get('usage'))
        calls.append({'response': out, 'real_body_ids': ids, 'clipped': clipped, 'message_index': len(messages)-1})
        for k in totals:
            totals[k] += (out.get('usage') or {}).get(k) or 0
        if done:
            finish = out['finish_reason']
            break
        if used >= budget['total_response_tokens']:
            break
        warned = g.warn(used, budget)
        messages.append({'role': 'user', 'content': B.CUT if warned else B.CONTINUE})
        if warned:
            left = g.stated(used, budget)
            messages = with_notice(messages, B.notice(left,final=left<=B.reserve(budget),extra=budget.get('notice_extra')))
    return {**out, 'content': '\n\n'.join(m.get('content') or '' for m in messages if m['role'] == 'assistant'),
            'reasoning': '\n\n'.join(m.get('reasoning_content') or '' for m in messages if m['role'] == 'assistant'),
            'finish_reason': finish, 'usage': totals,
            'response_budget': {'policy': B.POLICY, 'budget': budget, 'used': used, 'messages': messages, 'calls': calls,
                                'window': g.record()}}


async def run_async_chat(call, payload, budget):
    """As `run`, for asynchronous clients that return raw chat responses (HLE, SciCode): `call(payload)` is awaited
    and returns the OpenAI-format response dict."""
    payload = copy.deepcopy(payload)
    budget = B.item_budget(budget, payload.get('max_tokens'))
    requested = int(payload.get('max_tokens') or budget['total_response_tokens'])
    g = Window(budget, payload.get('max_tokens'))
    messages = payload['messages']
    used, calls, totals = 0, [], {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    finish = 'length'
    while used < budget['total_response_tokens'] and len(calls) < 16:
        cap = g.cap(used, budget)
        if cap < 1 and calls:   # the model's context room is spent: a length stop
            break
        payload['max_tokens'] = max(1, min(requested, cap))
        payload['messages'] = messages
        result = await call(request_payload(payload))
        msg, ids, clipped, used, done = step(payload, result, used, budget, messages)
        g.update(result.get('usage'))
        calls.append({'response': result, 'real_body_ids': ids, 'clipped': clipped, 'message_index': len(messages) - 1})
        for k in totals:
            totals[k] += (result.get('usage') or {}).get(k) or 0
        if done:
            finish = result['choices'][0]['finish_reason']
            break
        if used >= budget['total_response_tokens']:
            break
        warned = g.warn(used, budget)
        messages.append({'role': 'user', 'content': B.CUT if warned else B.CONTINUE})
        if warned:
            left = g.stated(used, budget)
            messages = with_notice(messages, B.notice(left, final=left <= B.reserve(budget),
                                                      extra=budget.get('notice_extra')))
    result = copy.deepcopy(calls[-1]['response'])
    reasoning = '\n\n'.join(m.get('reasoning_content') or '' for m in messages if m['role'] == 'assistant')
    content = '\n\n'.join(m.get('content') or '' for m in messages if m['role'] == 'assistant')
    result['choices'][0].update(message={'role': 'assistant', 'reasoning_content': reasoning, 'content': content},
                                finish_reason=finish)
    result['usage'] = totals
    result['_response_budget'] = {'policy': B.POLICY, 'budget': budget, 'used': used, 'messages': messages,
                                  'calls': calls, 'window': g.record()}
    return result
