"""The 128k context cap of the client (M3.2): /opt/aai/venvs/automationbench/bin/python -m
unittest aa_evals.automationbench.test_context_cap (no endpoint: the upstream request is stubbed)."""
import asyncio
import types
import unittest
from unittest import mock


class ContextCap(unittest.TestCase):
    def setUp(self):
        from automationbench.clients import RetryingOpenAIChatCompletionsClient
        from aa_evals.automationbench.run import make_client_class

        self.sent = []
        self.usage = (0, 0)

        async def fake(_self, prompt, model, sampling_args, tools=None, **kw):
            self.sent.append(dict(sampling_args))
            return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=self.usage[0],
                                                                     completion_tokens=self.usage[1]))

        self.patch = mock.patch.object(RetryingOpenAIChatCompletionsClient, 'get_native_response', fake)
        self.patch.start()
        cls = make_client_class()
        cls.context_window = 10_000
        self.client = cls.__new__(cls)   # no ClientConfig needed: only get_native_response is exercised

    def tearDown(self):
        self.patch.stop()

    def call(self, prompt, state, sampling=None):
        return asyncio.run(self.client.get_native_response(prompt, 'm', sampling or {}, None, state=state))

    def test_first_request_room_from_a_full_estimate(self):
        state = {}
        self.usage = (500, 100)
        self.call([{'role': 'user', 'content': 'x' * 2500}], state)
        self.assertLessEqual(self.sent[-1]['max_tokens'], 10_000 - 1000)
        self.assertEqual(state['_ctx_last'][2], 600)

    def test_incremental_estimate_uses_the_exact_previous_total(self):
        state = {}
        p = [{'role': 'user', 'content': 'hi'}]
        self.usage = (5000, 2000)
        self.call(p, state)
        p2 = p + [{'role': 'assistant', 'content': 'a'}, {'role': 'tool', 'content': 'y' * 250}]
        self.usage = (7200, 100)
        self.call(p2, state)
        room = self.sent[-1]['max_tokens']
        self.assertLess(room, 10_000 - 7000 - 100)      # at least 100 tokens for the 250 new characters
        self.assertGreater(room, 10_000 - 7000 - 200)

    def test_no_room_is_a_context_overflow(self):
        from verifiers.errors import OverlongPromptError
        state = {'_ctx_last': (1, '{"role": "user", "content": "hi"}', 9_900)}
        with self.assertRaises(OverlongPromptError):
            self.call([{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'a'},
                       {'role': 'tool', 'content': 'z'}], state)

    def test_a_smaller_requested_max_tokens_is_kept(self):
        self.usage = (10, 10)
        self.call([{'role': 'user', 'content': 'hi'}], {}, {'max_tokens': 300})
        self.assertEqual(self.sent[-1]['max_tokens'], 300)

    def test_prompt_over_the_cap_is_an_overflow(self):
        from verifiers.errors import OverlongPromptError
        self.usage = (10_500, 1)
        with self.assertRaises(OverlongPromptError):
            self.call([{'role': 'user', 'content': 'hi'}], {})


if __name__ == '__main__':
    unittest.main()
