"""snowflake-eval: the Snowflake (tiny GrugMoe) SFT checkpoints on snowball-eval's recipe with bf16 weights and KV.
The spec must differ from snowball-eval only by the FP8 flags, the slug / model / default checkpoint and the disk."""
import json
import tempfile
import unittest

from synth.config import load_teacher
from synth.vllm_launch import replica_command, replica_env
from task_synthesis.serve.serve_teacher import PROFILES

_TMP = tempfile.TemporaryDirectory()
FP8_FLAGS = {'--quantization': 'fp8', '--kv-cache-dtype': 'fp8'}


def launch(slug, checkpoint=None):
    teacher = load_teacher(slug, {'checkpoint': checkpoint} if checkpoint else None)
    profile = PROFILES[slug]
    tunables = {**teacher['serving'].get('tunables', {}), **profile.get('tunables', {})}
    argv = replica_command(teacher, profile['protocol'], port=20000, model_dir='/tmp/synth/model', tunables=tunables)
    env = replica_env(teacher, gpus=['0'], scratch=tempfile.mkdtemp(prefix=f'{slug}-test-', dir=_TMP.name),
                      tunables=tunables, base_env={'PATH': '/usr/bin:/bin'})
    return teacher, argv, env


def strip(argv, flags):
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a in flags:
            skip = True
            continue
        out.append(a)
    return out


class SnowflakeEval(unittest.TestCase):
    def test_bf16_weights_and_kv(self):
        _, argv, _ = launch('snowflake-eval')
        for name in FP8_FLAGS:
            self.assertNotIn(name, argv)
        self.assertEqual(json.loads(argv[argv.index('--hf-overrides') + 1]),
                         {'architectures': ['GrugMoeServeForCausalLM']})   # the plugin, never the in-tree model

    def test_same_recipe_as_snowball_otherwise(self):
        ck = '/data/aai/grug-b200/snowflake/tb/ckpts/m5l-sf-d512-full'   # {checkpoint} is expanded in the spec
        sb, a_sb, e_sb = launch('snowball-eval', ck)
        sf, a_sf, e_sf = launch('snowflake-eval', ck)
        i = a_sb.index('--hf-overrides')
        a_sb = a_sb[:i + 1] + [json.dumps({'architectures': ['GrugMoeServeForCausalLM']})] + a_sb[i + 2:]
        i = a_sf.index('--hf-overrides')
        a_sf = a_sf[:i + 1] + [json.dumps({'architectures': ['GrugMoeServeForCausalLM']})] + a_sf[i + 2:]
        self.assertEqual(strip(a_sb, FP8_FLAGS), a_sf)
        # OMP_NUM_THREADS follows each spec's CPU shape (snowball-eval 4 since the #531 trim; the Snowflakes 8)
        drop = lambda e: {k: v for k, v in e.items() if k != 'OMP_NUM_THREADS'
                          and not any(s in v for s in ('snowball-eval-test-', 'snowflake-eval-test-'))}
        self.assertEqual(e_sf.pop('VLLM_USE_FLASHINFER_MOE_FP16'), '0')   # Triton MoE for bf16 (spec header)
        self.assertEqual(drop(e_sb), drop(e_sf))
        self.assertEqual((e_sb['OMP_NUM_THREADS'], e_sf['OMP_NUM_THREADS']), ('4', '8'))
        rt = {**sf['runtime'], 'env': {k: v for k, v in sf['runtime']['env'].items() if k != 'VLLM_USE_FLASHINFER_MOE_FP16'}}
        self.assertEqual(sb['runtime'], rt)
        for key in ('served_name', 'request', 'checkpoint_checks', 'counting'):
            self.assertEqual(sb.get(key), sf.get(key), key)
        # the shape is sized per spec from its own measurements: snowball-eval was trimmed to its one-engine-per-GPU
        # use (#531); the Snowflakes run several engines per GPU (#460) and keep theirs until measured
        own = ('disk_gib', 'cpus_per_gpu', 'memory_gib_per_gpu', 'cpus_per_extra_engine')
        res_sb = {k: v for k, v in sb['serving_resources'].items() if k not in own}
        res_sf = {k: v for k, v in sf['serving_resources'].items() if k not in own}
        self.assertEqual(res_sb, res_sf)
        self.assertEqual(PROFILES['snowflake-eval'], PROFILES['snowball-eval'])


if __name__ == '__main__':
    unittest.main()
