"""Text-only eval variants (#348): each <slug>-text spec launches exactly its base's command, with
--language-model-only in place of the image allowance, under the same serve_teacher profile."""
import unittest

from synth.config import load_teacher
from synth.vllm_launch import replica_command, replica_env
from task_synthesis.serve.serve_teacher import PROFILES, TEXT_ONLY_VARIANTS

MM_FLAGS = ('--limit-mm-per-prompt', '--mm-processor-kwargs')
RANK_PORTS = {'dp_size': 2, 'rpc_port': 21000, 'supervisor_port': 22000}


def argv(slug):
    teacher = load_teacher(slug)
    profile = PROFILES[slug]
    tunables = {**teacher['serving'].get('tunables', {}), **profile.get('tunables', {})}
    return replica_command(teacher, profile.get('protocol', {'context_tokens': 131072, 'max_output_tokens': 131072}),
                           port=20000, model_dir='/tmp/m', tunables=tunables, rank_ports=RANK_PORTS)


def without_mm(args):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in MM_FLAGS:
            skip = True
            continue
        out.append(a)
    return out


class TextOnlyVariants(unittest.TestCase):
    def test_same_command_but_vision(self):
        for slug in TEXT_ONLY_VARIANTS:
            with self.subTest(slug=slug):
                base, text = argv(slug), argv(f'{slug}-text')
                self.assertIn('--limit-mm-per-prompt', base)
                self.assertNotIn('--language-model-only', base)
                self.assertEqual([a for a in text if a != '--language-model-only'], without_mm(base))
                self.assertEqual(text.count('--language-model-only'), 1)
                self.assertFalse(any(a in text for a in MM_FLAGS))

    def test_same_spec_and_profile_otherwise(self):
        for slug in TEXT_ONLY_VARIANTS:
            with self.subTest(slug=slug):
                base, text = load_teacher(slug), load_teacher(f'{slug}-text')
                self.assertEqual(text['slug'], f'{slug}-text')
                for key in set(base) | set(text):
                    if key not in ('slug', 'serving', 'serving_resources'):
                        self.assertEqual(text.get(key), base.get(key), key)
                # #378: without images a text-only server needs at most its base's host RAM and CPUs
                tr, br = text['serving_resources'], base['serving_resources']
                self.assertLessEqual(tr['memory_gib_per_gpu'], br['memory_gib_per_gpu'])
                self.assertLessEqual(tr['cpus_per_gpu'], br['cpus_per_gpu'])
                self.assertEqual({k: v for k, v in tr.items() if k not in ('memory_gib_per_gpu', 'cpus_per_gpu')},
                                 {k: v for k, v in br.items() if k not in ('memory_gib_per_gpu', 'cpus_per_gpu')})
                self.assertEqual({k: v for k, v in text['serving'].items() if k != 'args'},
                                 {k: v for k, v in base['serving'].items() if k != 'args'})
                self.assertIs(PROFILES[f'{slug}-text'], PROFILES[slug])
                env = dict(gpus=['0'], scratch='/tmp/synth-textonly-test', tunables={}, base_env={})
                self.assertEqual(replica_env(text, **env), replica_env(base, **env))


if __name__ == '__main__':
    unittest.main()
