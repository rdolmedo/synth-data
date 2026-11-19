"""Evaluator-only CIFAR-10 input adapter; never imported by an agent/setup hook.

The submitted network and its transforms remain the model under test. Only its
active external input sources are rebound to the pinned official test records.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ARCHIVE_SHA256 = 'c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd'
DESCRIPTOR_SHA256 = '82ad194b405494013cdde22a82a6f0c4acb6f3ed4f68ea62173ed76f43b9ea3f'
TEST_SHA256 = '8e2eb146ae340b09e24670f29cabc6326dba54da8789dab6768acf480273f65b'
RECORD_SIZE = 3073


class GraderAdapterInfrastructureError(RuntimeError):
    """The evaluator cannot safely adapt this legitimate input representation.

    This is explicitly not evidence of a bad submitted model. Native adoption
    requires a non-regenerating grading-pending handler for the emitted marker.
    """


class GraderAssetInfrastructureError(RuntimeError):
    """Evaluator-owned assets/runtime are unavailable or corrupt; not a model outcome."""

    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.evidence = evidence


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _verify_asset(path, expected):
    try:
        actual = sha256(path)
    except OSError as error:
        raise GraderAssetInfrastructureError('Evaluator asset unavailable: ' + str(path),
            {'path': str(path), 'expected_sha256': expected, 'actual_sha256': None}) from error
    if actual != expected:
        raise GraderAssetInfrastructureError('Evaluator asset hash mismatch: ' + str(path),
            {'path': str(path), 'expected_sha256': expected, 'actual_sha256': actual})


def message_types(descriptor):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    _verify_asset(descriptor, DESCRIPTOR_SHA256)
    files = descriptor_pb2.FileDescriptorSet.FromString(Path(descriptor).read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for item in files.file:
        pool.Add(item)
    return {name: message_factory.GetMessageClass(pool.FindMessageTypeByName('caffe.' + name))
            for name in ('NetParameter', 'Datum')}


def official_records(test_batch):
    # This is the exact member extracted from the hash-verified official archive.
    # No training batches are uploaded, and no model-owned data path is trusted.
    _verify_asset(test_batch, TEST_SHA256)
    with Path(test_batch).open('rb') as stream:
        for _ in range(10000):
            row = stream.read(RECORD_SIZE)
            if len(row) != RECORD_SIZE or row[0] > 9:
                raise GraderAssetInfrastructureError('Malformed official test record', {'path': str(test_batch), 'expected_sha256': TEST_SHA256, 'actual_sha256': sha256(test_batch)})
            yield row[0], row[1:]
        if stream.read(1):
            raise GraderAssetInfrastructureError('Official test batch has trailing bytes', {'path': str(test_batch), 'expected_sha256': TEST_SHA256, 'actual_sha256': sha256(test_batch)})


def make_lmdb(records, destination, datum_type):
    import lmdb
    destination.mkdir(parents=True, exist_ok=False)
    env = lmdb.open(str(destination), map_size=128 << 20)
    try:
        with env.begin(write=True) as txn:
            for i, (label, pixels) in enumerate(records):
                # Exactly BVLC's CIFAR converter: unencoded RGB CHW uint8 Datum.
                datum = datum_type(channels=3, height=32, width=32, label=label, data=pixels)
                txn.put(f'{i:05d}'.encode(), datum.SerializeToString())
        env.sync()
    finally:
        env.close()
    return destination


def make_image_list(records, destination):
    destination.mkdir(parents=True, exist_ok=False)
    lines = []
    for i, (label, pixels) in enumerate(records):
        # PPM stores RGB interleaved. Caffe ImageData applies its own unchanged
        # imread grayscale/BGR conversion, resize, and transform parameters.
        rgb = bytes(pixels[c * 1024 + p] for p in range(1024) for c in range(3))
        path = destination / f'{i:05d}.ppm'
        path.write_bytes(b'P6\n32 32\n255\n' + rgb)
        lines.append(f'{path} {label}\n')
    listing = destination / 'images.txt'
    listing.write_text(''.join(lines))
    return listing


def _rule_matches(rule, state):
    if rule.HasField('phase') and rule.phase != state.phase:
        return False
    if rule.HasField('min_level') and state.level < rule.min_level:
        return False
    if rule.HasField('max_level') and state.level > rule.max_level:
        return False
    return set(rule.stage) <= set(state.stage) and not (set(rule.not_stage) & set(state.stage))


def _active(layer, state):
    if layer.include and layer.exclude:
        raise GraderAdapterInfrastructureError('Layer has both include and exclude rules')
    return ((not layer.include or any(_rule_matches(r, state) for r in layer.include))
            and not any(_rule_matches(r, state) for r in layer.exclude))


def validate_data_representations(net, base, datum_type, evidence_dir, official_test_records):
    """Bound support to BVLC's raw uint8 RGB Datum, not hidden preconversions.

    Shape alone cannot establish RGB order or exclude external preprocessing.
    The first Datum must exactly match an already-private official TEST image.
    This is a bounded first-record domain check, not whole-database identity.
    Official records still replace the complete source regardless of this result.
    """
    import lmdb
    state = type(net.state)()
    state.CopyFrom(net.state)
    state.phase, state.level = 1, 0
    evidence_dir.mkdir(parents=True, exist_ok=True)
    official_rgb_pixels = None
    for layer in net.layer:
        if layer.type != 'Data' or not _active(layer, state):
            continue
        if layer.data_param.backend != 1:
            raise GraderAdapterInfrastructureError(f'Data representation needs a LevelDB adapter: {layer.name}')
        source = Path(layer.data_param.source)
        source = source if source.is_absolute() else base / source
        try:
            env = lmdb.open(str(source), readonly=True, lock=False, create=False)
            try:
                with env.begin() as txn:
                    cursor = txn.cursor()
                    if not cursor.first():
                        raise GraderAdapterInfrastructureError(f'Cannot inspect empty Data representation: {layer.name}')
                    raw = cursor.value()
            finally:
                env.close()
            datum = datum_type.FromString(raw)
        except GraderAdapterInfrastructureError:
            raise
        except Exception as error:
            raise GraderAdapterInfrastructureError(f'Cannot inspect Data representation: {layer.name}: {error}') from error
        digest = hashlib.sha256(raw).hexdigest()
        (evidence_dir / (digest + '.datum')).write_bytes(raw)
        details = {'layer': layer.name, 'source': str(source), 'first_datum_sha256': digest,
                   'encoded': datum.encoded, 'shape': [datum.channels, datum.height, datum.width],
                   'data_bytes': len(datum.data), 'float_values': len(datum.float_data)}
        canonical = (not datum.encoded and not datum.float_data
                     and (datum.channels, datum.height, datum.width) == (3, 32, 32)
                     and len(datum.data) == 3072)
        if canonical:
            if official_rgb_pixels is None:
                # In-memory lookup only: no new digest table or asset is staged.
                official_rgb_pixels = {pixels for _, pixels in official_test_records}
            pixels = datum.data
            swapped = pixels[2048:] + pixels[1024:2048] + pixels[:1024]
            details.update(official_test_sha256=TEST_SHA256,
                exact_official_rgb_match=pixels in official_rgb_pixels,
                distinct_swapped_official_match=(swapped != pixels and swapped in official_rgb_pixels))
        (evidence_dir / (digest + '.json')).write_text(json.dumps(details, indent=2) + '\n')
        if not canonical:
            raise GraderAdapterInfrastructureError('Unsupported Data representation: ' + json.dumps(details, sort_keys=True))
        if not details['exact_official_rgb_match'] or details['distinct_swapped_official_match']:
            raise GraderAdapterInfrastructureError('Unproven Data RGB pixel domain: ' + json.dumps(details, sort_keys=True))


def adapt_network(net, official_lmdb, official_images):
    """Return a copy with every active external input adapted, preserving transforms.

    TEST phase overrides net.state.phase, as caffe test does. Its default level is 0; submitted stages remain intact. Inactive TRAIN sources are never inspected/opened.
    """
    adapted = type(net)()
    adapted.CopyFrom(net)
    if net.layers or net.input:
        raise GraderAdapterInfrastructureError('Legacy net.layers/net.input requires an additional input adapter')
    state = type(net.state)()
    state.CopyFrom(net.state)
    state.phase = 1  # caffe.TEST
    state.level = 0  # caffe test default --level, overriding net.state.level
    found = []
    for index, layer in enumerate(net.layer):
        if not _active(layer, state):
            continue
        if layer.type in ('Data', 'ImageData'):
            if layer.bottom or not 1 <= len(layer.top) <= 2:
                raise GraderAdapterInfrastructureError(f'Unsupported {layer.type} input arity: {layer.name}')
            target = adapted.layer[index]
            if layer.type == 'Data':
                target.data_param.source = str(official_lmdb)
                target.data_param.backend = 1  # caffe.DataParameter.LMDB
            else:
                target.image_data_param.source = str(official_images)
                target.image_data_param.root_folder = ''
            found.append({'name': layer.name, 'type': layer.type, 'original_source':
                          layer.data_param.source if layer.type == 'Data' else layer.image_data_param.source})
        elif not layer.bottom:
            # Includes HDF5Data, MemoryData, Input, Python, DummyData, WindowData
            # and unknown/custom producers. Never retain a synthetic label source.
            raise GraderAdapterInfrastructureError(f'Unsupported active external input layer: {layer.name} ({layer.type})')
    if not found:
        raise GraderAdapterInfrastructureError('No supported active test input layer')
    return adapted, found


def capture_replay_inputs(base, net, out):
    """Capture referenced means outside normal artifact paths without running code.

    Task artifacts separately retain the complete examples/cifar10 directory,
    both build executable/library layouts, training log and build metadata.
    """
    out.mkdir(parents=True, exist_ok=True)
    means = []
    for layer in net.layer:
        if layer.transform_param.mean_file:
            path = Path(layer.transform_param.mean_file)
            path = path if path.is_absolute() else base / path
            if path.is_file():
                digest = sha256(path)
                dst = out / digest
                if not dst.exists():
                    shutil.copyfile(path, dst)
                means.append({'source': str(path), 'sha256': digest, 'captured': str(dst)})
            else:
                means.append({'source': str(path), 'missing': True})
    (out / 'mean-files.json').write_text(json.dumps(means, indent=2) + '\n')


def parse_test_accuracy(stderr):
    # Official tools/caffe.cpp prints "Batch i, accuracy = ..." for each of
    # 100 batches, then "accuracy = mean_score". Never grade the first batch.
    matches = re.findall(r'(?m)^(?:[^\n]*\]\s*)?accuracy = ([\d.]+)(?:\s|$)', stderr)
    assert matches, 'Could not find final aggregate accuracy in Caffe test output'
    return float(matches[-1])


def evaluate(base, assets, output, caffe_binary):
    from google.protobuf import text_format
    base, assets, output = Path(base), Path(assets), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    network = base / 'examples/cifar10/cifar10_quick_train_test.prototxt'
    net = None
    try:
        types = message_types(assets / 'caffe.desc')
        net = types['NetParameter']()
        try:
            text_format.Parse(network.read_text(), net)
        except text_format.ParseError as exc:
            raise AssertionError(f'Submitted network is not valid Caffe 1.0 prototxt: {exc}') from exc
        capture_replay_inputs(base, net, output / 'replay-means')
        # Validate every source before materializing any data or executing a model.
        adapted, sources = adapt_network(net, output / 'official-lmdb', output / 'official-images/images.txt')
        records = list(official_records(assets / 'test_batch.bin'))
        validate_data_representations(net, base, types['Datum'], output / 'input-representations', records)
    except GraderAdapterInfrastructureError as error:
        error.evidence = {
            'submitted_network_sha256': sha256(network), 'official_test_sha256': TEST_SHA256,
            'descriptor_sha256': DESCRIPTOR_SHA256, 'grader_sha256': sha256(__file__),
            'input_layers': [{'name': layer.name, 'type': layer.type,
                              'configuration_sha256': hashlib.sha256(layer.SerializeToString()).hexdigest()}
                             for layer in (net.layer if net is not None else []) if not layer.bottom],
        }
        raise
    if any(x['type'] == 'Data' for x in sources):
        make_lmdb(records, output / 'official-lmdb', types['Datum'])
    if any(x['type'] == 'ImageData' for x in sources):
        make_image_list(records, output / 'official-images')
    private_net = output / 'official-test.prototxt'
    private_net.write_text(text_format.MessageToString(adapted))
    (output / 'adapter.json').write_text(json.dumps({'schema_version': 1, 'sources': sources,
        'submitted_network_sha256': sha256(network), 'adapted_network_sha256': sha256(private_net),
        'official_archive_sha256': ARCHIVE_SHA256, 'official_test_sha256': TEST_SHA256, 'iterations': 100,
        'transform_policy': 'unchanged submitted layer parameters; external sources only'}, indent=2) + '\n')
    result = subprocess.run([str(caffe_binary), 'test', '-model', str(private_net), '-weights',
        str(base / 'examples/cifar10/cifar10_quick_iter_500.caffemodel'), '-iterations', '100'],
        capture_output=True, text=True, cwd=str(base))
    (output / 'test.stdout').write_text(result.stdout)
    (output / 'test.stderr').write_text(result.stderr)
    assert result.returncode == 0, f'Caffe test command failed: {result.stderr}'
    test_accuracy = parse_test_accuracy(result.stderr)
    training = (base / 'training_output.txt').read_text()
    matches = re.findall(r'Test net output #0: accuracy = ([\d.]+)', training)
    assert matches, 'Could not find any accuracy measurements in training output'
    # Historical label is misleading: this is the last TEST accuracy in the
    # training log, NOT training-set accuracy. Preserve it and its exact gap.
    return test_accuracy, float(matches[-1])


def record_adapter_infrastructure(error, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema_version': 1, 'kind': 'caffe_data_adapter_unsupported',
        'task': 'caffe-cifar-10', 'grading_pending': True, 'regenerate': False,
        'reason': str(error), 'evidence': error.evidence}, indent=2) + '\n')


def record_asset_infrastructure(error, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema_version': 1, 'kind': 'caffe_grader_asset_invalid',
        'task': 'caffe-cifar-10', 'reason': str(error), 'evidence': error.evidence}, indent=2) + '\n')
