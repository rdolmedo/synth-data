from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from google.protobuf import text_format
import lmdb

from aa_evals.tb21.caffe_repair import bundle, grader

ASSETS = Path(os.environ.get('CAFFE_GRADER_TEST_ASSETS', '/data/aai/aa_evals/grader_assets/cifar10/v1'))


class OfficialAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.types = grader.message_types(ASSETS / 'caffe.desc')
        cls.official = list(grader.official_records(ASSETS / 'test_batch.bin'))

    def net(self, text):
        net = self.types['NetParameter']()
        text_format.Parse(text, net)
        return net

    def test_all_active_inputs_rebound_and_legitimate_transforms_unchanged(self):
        net = self.net('''state { level: 7 stage: "eval" }
          layer { name:"train" type:"HDF5Data" top:"unused" include { phase: TRAIN } }
          layer { name:"first" type:"Data" top:"data" top:"label"
            data_param { source:"synthetic-lmdb" batch_size:100 backend:LEVELDB rand_skip:3 }
            transform_param { scale:0.0039 mirror:true crop_size:28 mean_value:127 mean_value:120 mean_value:110 } }
          layer { name:"second" type:"ImageData" top:"data2" top:"label2"
            image_data_param { source:"synthetic-list" root_folder:"/old/" batch_size:100
              new_height:30 new_width:30 is_color:false shuffle:true }
            transform_param { mean_file:"examples/cifar10/mean.binaryproto" scale:0.5 } }
          layer { name:"downstream" type:"InnerProduct" bottom:"data" top:"scores"
            inner_product_param { num_output:10 } }''')
        before = net.SerializeToString()
        adapted, sources = grader.adapt_network(net, Path('/private/official'), Path('/private/list'))
        self.assertEqual(net.SerializeToString(), before)
        self.assertEqual(len(sources), 2)
        expected = self.types['NetParameter']()
        expected.CopyFrom(net)
        expected.layer[1].data_param.source = '/private/official'
        expected.layer[1].data_param.backend = 1
        expected.layer[2].image_data_param.source = '/private/list'
        expected.layer[2].image_data_param.root_folder = ''
        self.assertEqual(adapted.SerializeToString(), expected.SerializeToString())

    def test_actual_caffe_cli_level_zero_phase_and_stages(self):
        net = self.net('''state { level:7 phase:TRAIN stage:"eval" }
          layer { name:"data" type:"Data" top:"data" include { phase:TEST max_level:0 stage:"eval" }
            data_param { source:"old" batch_size:100 } }
          layer { name:"inactive" type:"MemoryData" top:"unused" include { min_level:1 } }''')
        adapted, sources = grader.adapt_network(net, '/private/db', '/private/list')
        self.assertEqual([s['name'] for s in sources], ['data'])
        self.assertEqual(adapted.layer[0].data_param.source, '/private/db')
        self.assertEqual(adapted.layer[1].SerializeToString(), net.layer[1].SerializeToString())

    def test_unsupported_legitimate_inputs_are_typed_grader_infrastructure(self):
        for kind in ('HDF5Data', 'MemoryData', 'Input', 'WindowData', 'Python', 'DummyData'):
            with self.subTest(kind=kind):
                net = self.net(f'layer {{ name:"input" type:"{kind}" top:"data" }}')
                with self.assertRaises(grader.GraderAdapterInfrastructureError):
                    grader.adapt_network(net, '/private/db', '/private/list')
        # A second synthetic/unsupported producer must not sneak through after
        # one valid official-data producer has already been found.
        net = self.net('''layer { name:"d" type:"Data" top:"data" data_param { batch_size:100 } }
                          layer { name:"labels" type:"DummyData" top:"label" }''')
        with self.assertRaises(grader.GraderAdapterInfrastructureError):
            grader.adapt_network(net, '/private/db', '/private/list')

    def test_data_representation_standard_positive_and_noncanonical_pending(self):
        variants = [
            ('standard', dict(channels=3, height=32, width=32, data=self.official[0][1]), True),
            ('encoded', dict(channels=3, height=32, width=32, data=b'encoded image', encoded=True), False),
            ('float', dict(channels=3, height=32, width=32, float_data=[0.5] * 3072), False),
            ('wrong-shape', dict(channels=1, height=32, width=32, data=bytes(1024)), False),
        ]
        for name, fields, supported in variants:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                db = root / 'original-db'
                env = lmdb.open(str(db), map_size=1 << 20)
                raw = self.types['Datum'](label=0, **fields).SerializeToString()
                with env.begin(write=True) as txn:
                    txn.put(b'00000', raw)
                env.close()
                before = {p.name: grader.sha256(p) for p in db.iterdir()}
                net = self.net('layer { name:"input" type:"Data" top:"data" data_param { source:"original-db" backend:LMDB } }')
                if supported:
                    grader.validate_data_representations(net, root, self.types['Datum'], root / 'evidence', self.official)
                else:
                    with self.assertRaises(grader.GraderAdapterInfrastructureError):
                        grader.validate_data_representations(net, root, self.types['Datum'], root / 'evidence', self.official)
                self.assertEqual(before, {p.name: grader.sha256(p) for p in db.iterdir()})
                self.assertEqual((root / 'evidence' / (hashlib.sha256(raw).hexdigest() + '.datum')).read_bytes(), raw)

    def test_rgb_bgr_unknown_and_swapped_equivalence_domain_guards(self):
        rgb = self.official[0][1]
        bgr = rgb[2048:] + rgb[1024:2048] + rgb[:1024]
        official_pixels = {pixels for _, pixels in self.official}
        self.assertNotEqual(rgb, bgr)
        self.assertNotIn(bgr, official_pixels)
        self.assertNotIn(bytes(3072), official_pixels)
        equivalent = bytes([7]) * 1024 + bytes([15]) * 1024 + bytes([7]) * 1024
        variants = [
            ('actual-official-rgb', rgb, self.official, True),
            ('bgr', bgr, self.official, False),
            ('unrecognized', bytes(3072), self.official, False),
            # Synthetic reference pairs isolate the ambiguity rule; they are
            # unit fixtures only, never production assets or uploaded tables.
            ('distinct-swapped-both-match', rgb, [(0, rgb), (0, bgr)], False),
            ('identical-rb-swap-equivalent', equivalent, [(0, equivalent)], True),
        ]
        for name, pixels, references, supported in variants:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                grader.make_lmdb([(0, pixels)], root / 'source', self.types['Datum'])
                net = self.net('layer { name:"input" type:"Data" top:"data" data_param { source:"source" backend:LMDB } }')
                before = {p.name: grader.sha256(p) for p in (root / 'source').iterdir()}
                if supported:
                    grader.validate_data_representations(net, root, self.types['Datum'], root / 'evidence', references)
                else:
                    with self.assertRaisesRegex(grader.GraderAdapterInfrastructureError, 'Unproven Data RGB'):
                        grader.validate_data_representations(net, root, self.types['Datum'], root / 'evidence', references)
                self.assertEqual(before, {p.name: grader.sha256(p) for p in (root / 'source').iterdir()})

    def test_bgr_source_is_pending_before_any_model_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = root / 'examples/cifar10'
            examples.mkdir(parents=True)
            rgb = self.official[0][1]
            bgr = rgb[2048:] + rgb[1024:2048] + rgb[:1024]
            grader.make_lmdb([(0, bgr)], root / 'source', self.types['Datum'])
            (examples / 'cifar10_quick_train_test.prototxt').write_text(
                'layer { name:"input" type:"Data" top:"data" top:"label" data_param { source:"source" batch_size:100 backend:LMDB } }')
            with patch.object(grader.subprocess, 'run') as execute:
                with self.assertRaises(grader.GraderAdapterInfrastructureError) as caught:
                    grader.evaluate(root, ASSETS, root / 'output', '/not-executed')
                execute.assert_not_called()
            marker = root / 'verifier/grading-pending.json'
            grader.record_adapter_infrastructure(caught.exception, marker)
            self.assertIs(json.loads(marker.read_text())['grading_pending'], True)
            self.assertIs(json.loads(marker.read_text())['regenerate'], False)

    def test_official_records_and_lmdb_are_exact_rgb_chw_and_labels(self):
        records = list(grader.official_records(ASSETS / 'test_batch.bin'))
        self.assertEqual(len(records), 10000)
        self.assertEqual([sum(label == n for label, _ in records) for n in range(10)], [1000] * 10)
        with tempfile.TemporaryDirectory() as tmp:
            path = grader.make_lmdb(records, Path(tmp) / 'lmdb', self.types['Datum'])
            env = lmdb.open(str(path), readonly=True, lock=False)
            with env.begin() as txn:
                for index in (0, 1, 9999):
                    datum = self.types['Datum'].FromString(txn.get(f'{index:05d}'.encode()))
                    self.assertEqual((datum.label, datum.data), records[index])
                    self.assertEqual((datum.channels, datum.height, datum.width), (3, 32, 32))
            env.close()

    def test_synthetic_test_batch_with_official_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'test_batch.bin'
            fake.write_bytes(b'synthetic model-owned substitute')
            with self.assertRaisesRegex(grader.GraderAssetInfrastructureError, 'hash mismatch'):
                list(grader.official_records(fake))

    def test_synthetic_perfect_score_cannot_use_submitted_data(self):
        """Neutral constant-label classifier gets .1 on official labels, not 1.

        Exercise the actual evaluator adapter and command arguments, without
        running a saved checkpoint or implementing a task model/training loop.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base, output = Path(tmp) / 'app', Path(tmp) / 'grader'
            examples = base / 'examples/cifar10'
            examples.mkdir(parents=True)
            net = self.net('''layer { name:"input" type:"Data" top:"data" top:"label"
              data_param { source:"synthetic-all-class-zero" batch_size:100 backend:LMDB } }''')
            # One exact official RGB anchor proves first-record representation;
            # all model-owned labels/remaining synthetic records are still replaced.
            grader.make_lmdb([(0, self.official[0][1])] + [(0, bytes(3072))] * 99,
                             base / 'synthetic-all-class-zero', self.types['Datum'])
            (examples / 'cifar10_quick_train_test.prototxt').write_text(text_format.MessageToString(net))
            (base / 'training_output.txt').write_text('Test net output #0: accuracy = 1.0\n')
            def constant_label_model(argv, **kwargs):
                adapted = self.net(Path(argv[argv.index('-model') + 1]).read_text())
                self.assertEqual(argv[-2:], ['-iterations', '100'])
                source = Path(adapted.layer[0].data_param.source)
                self.assertNotIn('synthetic', str(source))
                env = lmdb.open(str(source), readonly=True, lock=False)
                with env.begin() as txn:
                    labels = [self.types['Datum'].FromString(value).label for _, value in txn.cursor()]
                env.close()
                accuracy = labels.count(0) / len(labels)
                return SimpleNamespace(returncode=0, stdout='', stderr=f'accuracy = {accuracy}\n')
            with patch.object(grader.subprocess, 'run', constant_label_model):
                test_accuracy, comparator = grader.evaluate(base, ASSETS, output, '/neutral/constant-label-model')
            self.assertEqual(comparator, 1.0)
            self.assertEqual(test_accuracy, 0.1)
            self.assertFalse(test_accuracy > 0.45)

    def test_image_list_preserves_color_data_and_labels(self):
        pixels = bytes([11] * 1024 + [22] * 1024 + [33] * 1024)
        with tempfile.TemporaryDirectory() as tmp:
            listing = grader.make_image_list([(8, pixels)], Path(tmp) / 'images')
            image = listing.parent / '00000.ppm'
            self.assertEqual(image.read_bytes(), b'P6\n32 32\n255\n' + bytes([11, 22, 33]) * 1024)
            self.assertEqual(listing.read_text(), f'{image} 8\n')

    def test_external_mean_is_captured_without_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mean = root / 'external-mean.binaryproto'
            mean.write_bytes(b'unchanged model-selected mean')
            net = self.net(f'layer {{ name:"d" type:"Data" top:"data" transform_param {{ mean_file:"{mean}" }} }}')
            grader.capture_replay_inputs(root / 'app', net, root / 'replay')
            captured = json.loads((root / 'replay/mean-files.json').read_text())[0]
            self.assertEqual(Path(captured['captured']).read_bytes(), mean.read_bytes())
            self.assertEqual(captured['sha256'], hashlib.sha256(mean.read_bytes()).hexdigest())

    def test_original_other_tests_metrics_and_task_environment_unchanged(self):
        import tarfile
        with tarfile.open('/data/aai/tbenchtrain/eval_q35/bundle/tb21_6h.tar') as tf:
            source = tf.extractfile(bundle.TASK + 'tests/test_outputs.py').read()
            config = tf.extractfile(bundle.TASK + 'task.toml').read()
        patched = bundle.patch_test(source)
        self.assertIn(b'test_accuracy > 0.45', patched)
        self.assertIn(b'accuracy_diff <= 0.05', patched)
        new_config = bundle.patch_config(config)
        import tomllib
        self.assertEqual(tomllib.loads(new_config.decode())['agent'], tomllib.loads(config.decode())['agent'])
        self.assertEqual(tomllib.loads(new_config.decode())['environment'], tomllib.loads(config.decode())['environment'])

    def test_typed_pending_marker_is_bound_and_never_runs_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'app'
            examples = base / 'examples/cifar10'
            examples.mkdir(parents=True)
            network = examples / 'cifar10_quick_train_test.prototxt'
            network.write_text('layer { name:"hdf5" type:"HDF5Data" top:"data" top:"label" }')
            with patch.object(grader.subprocess, 'run') as execute:
                try:
                    grader.evaluate(base, ASSETS, Path(tmp) / 'output', '/not-executed')
                except grader.GraderAdapterInfrastructureError as error:
                    marker = Path(tmp) / 'verifier/grading-pending.json'
                    grader.record_adapter_infrastructure(error, marker)
                else:
                    self.fail('Unsupported source must not get an accuracy verdict')
                execute.assert_not_called()
            value = json.loads(marker.read_text())
            self.assertIs(type(value['schema_version']), int)
            self.assertEqual(value['kind'], 'caffe_data_adapter_unsupported')
            self.assertEqual(value['task'], 'caffe-cifar-10')
            self.assertIs(value['grading_pending'], True)
            self.assertIs(value['regenerate'], False)
            self.assertEqual(value['evidence']['submitted_network_sha256'], grader.sha256(network))
            self.assertEqual(value['evidence']['input_layers'][0]['type'], 'HDF5Data')
            self.assertEqual(value['evidence']['grader_sha256'], grader.sha256(grader.__file__))

    def test_shell_clears_stale_marker_and_withholds_numeric_pending_reward(self):
        import subprocess
        import tarfile
        with tarfile.open('/data/aai/tbenchtrain/eval_q35/bundle/tb21_6h.tar') as tf:
            original = tf.extractfile(bundle.TASK + 'tests/test.sh').read()
        for mode, exitcode, expected in [('normal', '0', '1'), ('normal', '1', '0'),
                                          ('valid', '1', None), ('malformed', '1', None), ('asset', '1', None)]:
            with self.subTest(mode=mode, exitcode=exitcode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                logs = root / 'logs'
                logs.mkdir()
                (logs / 'grading-pending.json').write_text('stale')
                (logs / 'caffe-grader-asset-infrastructure.json').write_text('stale')
                (logs / 'reward.txt').write_text('stale')
                (logs / 'reward.json').write_text('stale')
                bins = root / '.local/bin'
                bins.mkdir(parents=True)
                (bins / 'env').write_text('')
                for name in ('apt-get', 'curl'):
                    p = bins / name
                    p.write_text('#!/bin/sh\nexit 0\n')
                    p.chmod(0o755)
                uvx = bins / 'uvx'
                uvx.write_text('''#!/bin/sh
[ ! -e "$TEST_LOGS/grading-pending.json" ] || exit 88
if [ "$MODE" = valid ]; then echo '{"schema_version":1,"kind":"caffe_data_adapter_unsupported"}' > "$TEST_LOGS/grading-pending.json"; fi
if [ "$MODE" = malformed ]; then echo malformed > "$TEST_LOGS/grading-pending.json"; fi
if [ "$MODE" = asset ]; then echo asset > "$TEST_LOGS/caffe-grader-asset-infrastructure.json"; fi
exit "$TEST_EXIT"
''')
                uvx.chmod(0o755)
                script = root / 'test.sh'
                script.write_bytes(bundle.patch_script(original).replace(b'/logs/verifier', str(logs).encode()))
                env = dict(os.environ, HOME=str(root), PATH=str(bins) + ':' + os.environ['PATH'],
                           TEST_LOGS=str(logs), MODE=mode, TEST_EXIT=exitcode)
                result = subprocess.run(['bash', str(script)], cwd=root, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse((logs / 'reward.json').exists())
                if expected is None:
                    self.assertFalse((logs / 'reward.txt').exists())
                else:
                    self.assertEqual((logs / 'reward.txt').read_text().strip(), expected)
                    self.assertFalse((logs / 'grading-pending.json').exists())

    def test_realistic_caffe_output_uses_final_aggregate_not_first_batch(self):
        stderr = ("I1107 00:00:00.000000 42 caffe.cpp:310] Batch 0, accuracy = 0.99\n"
                  "I1107 00:00:00.000001 42 caffe.cpp:310] Batch 1, accuracy = 0.01\n"
                  "I1107 00:00:00.000002 42 caffe.cpp:318] Loss: 1.5\n"
                  "I1107 00:00:00.000003 42 caffe.cpp:329] accuracy = 0.40\n"
                  "I1107 00:00:00.000004 42 caffe.cpp:329] loss = 1.5 (* 1 = 1.5 loss)\n")
        self.assertEqual(grader.parse_test_accuracy(stderr), 0.40)
        self.assertFalse(grader.parse_test_accuracy(stderr) > 0.45)
        with self.assertRaisesRegex(AssertionError, 'final aggregate'):
            grader.parse_test_accuracy('I1107 caffe.cpp:310] Batch 0, accuracy = 0.99\n')

    def test_asset_failure_never_becomes_unsupported_model_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = Path(tmp) / 'assets'
            assets.mkdir()
            (assets / 'caffe.desc').write_bytes(b'corrupt evaluator descriptor')
            with self.assertRaises(grader.GraderAssetInfrastructureError) as caught:
                grader.evaluate(Path(tmp) / 'app', assets, Path(tmp) / 'out', '/not-executed')
            marker = Path(tmp) / 'verifier/caffe-grader-asset-infrastructure.json'
            grader.record_asset_infrastructure(caught.exception, marker)
            value = json.loads(marker.read_text())
            self.assertEqual(value['kind'], 'caffe_grader_asset_invalid')
            self.assertNotIn('grading_pending', value)
            self.assertFalse((marker.parent / 'grading-pending.json').exists())
            self.assertEqual(value['evidence']['expected_sha256'], grader.DESCRIPTOR_SHA256)
            self.assertEqual(value['evidence']['actual_sha256'], grader.sha256(assets / 'caffe.desc'))

    def test_actual_harbor_upload_boundary_after_agent(self):
        # Run installed Harbor's real phase-ordering and real tests-upload code
        # against neutral fake callbacks; no environment, model, or network.
        from harbor.trial.single_step import SingleStepTrial
        from harbor.verifier.verifier import Verifier
        events = []
        class Uploaded(Exception): pass
        async def agent(): events.append('agent-finished')
        async def logs(): pass
        async def collect(**kwargs): events.append('artifacts')
        async def stop(): pass
        async def upload(**kwargs):
            self.assertEqual(events, ['agent-finished', 'artifacts'])
            events.append('tests-uploaded')
            raise Uploaded()
        with tempfile.TemporaryDirectory() as tmp:
            tests = Path(tmp) / 'tests'
            tests.mkdir()
            (tests / 'test.sh').write_text('true\n')
            (tests / 'official-assets').mkdir()
            (tests / 'official-assets/private-test-marker').write_text('evaluator-only')
            env = SimpleNamespace(os='linux', upload_dir=upload)
            verifier = SimpleNamespace(environment=env, _skip_tests_upload=False,
                _resolve_tests=lambda: ([tests], tests, tests / 'test.sh'),
                task=SimpleNamespace(paths=SimpleNamespace(validate_input_tree=lambda p: None)))
            async def verify():
                try:
                    await Verifier.verify(verifier)
                except Exception as error:
                    # Native verifier wraps upload failure; our sentinel is its cause.
                    if not isinstance(error.__cause__, Uploaded):
                        raise
            from harbor.models.task.config import TaskConfig
            fake = SimpleNamespace(task=SimpleNamespace(config=TaskConfig()),
                _run_agent=agent, _upload_agent_logs=logs, _collect_artifacts=collect,
                _run_verifier=verify, _stop_agent_environment=stop)
            asyncio.run(SingleStepTrial._run(fake))
        self.assertEqual(events, ['agent-finished', 'artifacts', 'tests-uploaded'])


if __name__ == '__main__':
    unittest.main()
