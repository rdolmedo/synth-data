"""replicas mode with tunable engines_per_gpu: k engines per GPU group, each with gpu_util / k."""
import unittest

from task_synthesis.serve.serve_teacher import engine_tunables, replica_groups


class EnginesPerGpu(unittest.TestCase):
    def test_default_is_one_engine_per_group(self):
        groups, k = replica_groups(['0', '1'], {'gpus_per_replica': 1}, {'gpu_util': 0.9})
        self.assertEqual((groups, k), ([['0'], ['1']], 1))
        self.assertEqual(engine_tunables({'gpu_util': 0.9}, 1), {'gpu_util': 0.9})

    def test_k_engines_share_each_gpu(self):
        groups, k = replica_groups(['0'], {}, {'gpu_util': 0.9, 'engines_per_gpu': 3})
        self.assertEqual((groups, k), ([['0'], ['0'], ['0']], 3))
        self.assertEqual(engine_tunables({'gpu_util': 0.9, 'max_seqs': 256}, 3), {'gpu_util': 0.3, 'max_seqs': 256})

    def test_multi_gpu_groups_repeat_in_order(self):
        groups, _ = replica_groups(['0', '1', '2', '3'], {'gpus_per_replica': 2}, {'engines_per_gpu': 2})
        self.assertEqual(groups, [['0', '1'], ['0', '1'], ['2', '3'], ['2', '3']])


if __name__ == '__main__':
    unittest.main()
