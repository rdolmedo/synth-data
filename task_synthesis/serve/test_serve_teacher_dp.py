"""serve_teacher's dp_external_lb plan: one engine by default, `dp_group_gpus` engines with disjoint ports otherwise."""
import itertools
import unittest

from synth.config import load_teacher
from synth.vllm_launch import replica_command
from task_synthesis.serve.serve_teacher import PROFILES, dp_external_plan


def ports_from(*bases):
    it = iter(bases)
    return lambda n: next(it)


class Plan(unittest.TestCase):
    def test_one_engine_without_group_size(self):
        plan = dp_external_plan(['a', 'b', 'c', 'd'], {}, ports_from(20000, 21000, 22000))
        self.assertEqual(plan, [(['a', 'b', 'c', 'd'], 20000, {'dp_size': 4, 'rpc_port': 21000, 'supervisor_port': 22000})])

    def test_two_dp2_engines(self):
        plan = dp_external_plan(['a', 'b', 'c', 'd'], {'dp_group_gpus': 2},
                                ports_from(20000, 21000, 22000, 23000, 24000, 25000))
        self.assertEqual(plan, [(['a', 'b'], 20000, {'dp_size': 2, 'rpc_port': 21000, 'supervisor_port': 22000}),
                                (['c', 'd'], 23000, {'dp_size': 2, 'rpc_port': 24000, 'supervisor_port': 25000})])

    def test_overlapping_ports_are_redrawn(self):
        # engine 2's API block would overlap engine 1's rank ports (20000-20001): redrawn
        plan = dp_external_plan(['a', 'b', 'c', 'd'], {'dp_group_gpus': 2},
                                ports_from(20000, 21000, 22000, 20001, 23000, 24000, 25000))
        self.assertEqual(plan[1], (['c', 'd'], 23000, {'dp_size': 2, 'rpc_port': 24000, 'supervisor_port': 25000}))

    def test_group_size_must_divide(self):
        with self.assertRaises(ValueError):
            dp_external_plan(['a', 'b', 'c', 'd'], {'dp_group_gpus': 3}, itertools.count(20000).__next__)


class GlmEvalTeacher(unittest.TestCase):
    def test_argv_is_the_study_recipe_per_engine(self):
        teacher = load_teacher('glm53-flash-eval')
        self.assertEqual(teacher['serving']['dp_group_gpus'], 2)
        self.assertNotIn('env', teacher['request'])   # module absent from the frozen eval code
        profile = PROFILES['glm53-flash-eval']
        tunables = {**teacher['serving']['tunables'], **profile['tunables']}
        extra = {'dp_size': 2, 'rpc_port': 21000, 'supervisor_port': 22000}
        argv = replica_command(teacher, profile['protocol'], port=20000, model_dir='/tmp/m', tunables=tunables,
                               rank_ports=extra)
        text = ' '.join(argv)
        for want in ('--data-parallel-size 2', '--data-parallel-size-local 2', '--data-parallel-rpc-port 21000',
                     '--data-parallel-supervisor-port 22000', '--max-model-len 131072', '--max-num-seqs 256',
                     '--max-num-batched-tokens 8192', '--gpu-memory-utilization 0.95',
                     '--mamba-ssm-cache-dtype bfloat16', '--kv-offloading-size 64', '--enable-prompt-tokens-details',
                     '"num_speculative_tokens":3', '"max_new_tokens": 131072',
                     '--limit-mm-per-prompt {"image":256,"video":0}'):
            self.assertIn(want, text)


if __name__ == '__main__':
    unittest.main()
