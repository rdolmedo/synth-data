"""Teacher token-counting serializations (the only teacher-specific export code).

The exporter converts native Pi events to OpenAI-style messages (reasoning in
`reasoning_content`, tool-call arguments as dicts, `_stop_reason` kept for
encoders that mark unterminated turns) and asks the teacher's serializer to
render them. Counting uses the pinned tokenizer of the served checkpoint.
"""
import copy
import functools
import importlib.util
import json
from pathlib import Path

from .common import sha


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def public(message):
    return {k: v for k, v in message.items() if not k.startswith('_') and k not in (
        'native_omitted_attachments', 'attachment_accounting')}


class JinjaSerializer:
    """Publisher chat template rendered in a sandbox with fixed kwargs."""

    def __init__(self, teacher):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        counting = teacher['counting']
        self.tokenizer_path = Path(counting['tokenizer'])
        self.template_path = Path(counting['chat_template'])
        self.kwargs = counting.get('render_kwargs', {})
        self.crosscheck = counting.get('tokenize_crosscheck', 'native')
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
        env.filters['tojson'] = tojson
        self.template = env.from_string(self.template_path.read_text())

    def render(self, messages, tools):
        return self.template.render(messages=[public(m) for m in messages], tools=tools, **self.kwargs)

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    @functools.cache
    def identity(self):  # hashed once per process: the tokenizer file alone took ~11% of an export
        return {'serializer': 'jinja', 'tokenizer_sha256': sha(self.tokenizer_path),
                'chat_template_sha256': sha(self.template_path), 'render_kwargs': self.kwargs}

    def tokenize_payload(self, messages, teacher):
        """/tokenize request that must reproduce count(render(messages))."""
        if self.crosscheck in (None, 'none'):
            return None
        wire = [dict(public(m)) for m in messages]
        for m in wire:
            reasoning = m.pop('reasoning_content', None)
            if not reasoning:
                continue
            if self.crosscheck == 'inline_think':  # this runtime's /tokenize drops reasoning fields
                m['content'] = '<think>\n' + reasoning + '</think>' + (m.get('content') or '')
            elif self.crosscheck == 'reasoning_field':
                m['reasoning'] = reasoning
            else:
                m['reasoning_content'] = reasoning
        kwargs = {k: v for k, v in self.kwargs.items() if k != 'add_generation_prompt'}
        return {'model': teacher['served_name'], 'messages': wire,
                'add_generation_prompt': self.kwargs.get('add_generation_prompt', False),
                'chat_template_kwargs': kwargs}


class DeepSeekSerializer:
    """DeepSeek V4/V4.1 ship no chat template; vLLM's encoder module is the reference."""

    def __init__(self, teacher):
        from tokenizers import Tokenizer
        counting = teacher['counting']
        self.encoder_path = Path(counting['encoder'])
        spec = importlib.util.spec_from_file_location(f'pinned_{teacher["slug"].replace("-", "_")}_encoding',
                                                      self.encoder_path)
        self.encoding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.encoding)
        self.tokenizer_path = Path(counting['tokenizer'])
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.kwargs = counting.get('encode_kwargs', {'thinking_mode': 'thinking', 'drop_thinking': False,
                                                     'reasoning_effort': 'max'})

    def adapt(self, messages, tools):
        out = []
        for m in messages:
            m = copy.deepcopy(public(m))
            if m['role'] == 'system' and tools and not any('tools' in x for x in out):
                m['tools'] = tools
            if m['role'] == 'assistant':
                m['reasoning'] = m.get('reasoning_content', '')
                for call in m.get('tool_calls', []):
                    args = call['function']['arguments']
                    if not isinstance(args, str):
                        call['function']['arguments'] = json.dumps(args, ensure_ascii=False, separators=(',', ':'))
            out.append(m)
        for original, m in zip(messages, out):
            if original.get('_stop_reason') in ('length', 'error', 'aborted'):
                m['wo_eos'] = True
        return out

    def render(self, messages, tools):
        return self.encoding.encode_messages(self.adapt(messages, tools), context=None, **self.kwargs)

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    @functools.cache
    def identity(self):
        return {'serializer': 'deepseek', 'encoder_sha256': sha(self.encoder_path),
                'tokenizer_sha256': sha(self.tokenizer_path), 'encode_kwargs': self.kwargs}

    def tokenize_payload(self, messages, teacher):
        return None  # vLLM /tokenize crashes on DeepSeek tool histories (deepseek-v41 campaign)


def serializer_for(teacher):
    kind = teacher['counting'].get('serializer', 'jinja')
    return {'jinja': JinjaSerializer, 'deepseek': DeepSeekSerializer}[kind](teacher)
