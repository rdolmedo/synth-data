"""Build a new Caffe-only grader overlay; task content and assets stay outside git.

Usage: python -m aa_evals.tb21.caffe_repair.bundle OLD.tar ASSET_DIR NEW.tar
Output is exclusive-create. No existing bundle/task directory is modified.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import sys
import tarfile
import tomllib
from pathlib import Path

from . import grader

TASK = 'tasks/caffe-cifar-10/'
ORIGINAL_BUNDLE_SHA256 = '363d5b9344a87969180a253a489404bf2fd3d0a550006b92609340dbbb538c50'
ARTIFACTS = [
    {'source': '/app/caffe/examples/cifar10', 'exclude': ['*lmdb*', '*leveldb*']},
    {'source': '/app/caffe/training_output.txt'},
    {'source': '/app/caffe/.build_release/tools/caffe.bin'},
    {'source': '/app/caffe/.build_release/lib', 'exclude': ['*.a', '*.o', '*.d']},
    {'source': '/app/caffe/build/tools/caffe'},
    {'source': '/app/caffe/build/lib', 'exclude': ['*.a', '*.o', '*.d']},
    {'source': '/app/caffe/Makefile.config'},
    {'source': '/app/caffe/build/CMakeCache.txt'},
]
FIXED_ARTIFACTS = [{'source': path} for path in (
    '/app/caffe/examples/cifar10/cifar10_quick_train_test.prototxt',
    '/app/caffe/examples/cifar10/cifar10_quick_solver.prototxt',
    '/app/caffe/examples/cifar10/cifar10_quick_iter_500.caffemodel',
    '/app/caffe/training_output.txt',
)]

REPLACEMENT = '''def test_model_accuracy_verification():
    """Evaluate the submitted model on evaluator-owned official CIFAR-10 inputs."""
    from caffe_official_grader import (
        evaluate, GraderAdapterInfrastructureError, record_adapter_infrastructure,
        GraderAssetInfrastructureError, record_asset_infrastructure,
    )
    try:
        test_accuracy, final_training_accuracy = evaluate(
            Path("/app/caffe"), Path("/tests/official-assets"),
            Path("/logs/verifier/official-cifar10"), _find_caffe_binary(),
        )
    except GraderAssetInfrastructureError as error:
        record_asset_infrastructure(error, "/logs/verifier/caffe-grader-asset-infrastructure.json")
        raise
    except GraderAdapterInfrastructureError as error:
        record_adapter_infrastructure(error, "/logs/verifier/grading-pending.json")
        raise
    # Preserve the original thresholds and comparator. This historical variable
    # is the last TEST-net accuracy logged during training, not train accuracy.
    assert test_accuracy > 0.45, f"Model accuracy {test_accuracy} is below 45% threshold"
    accuracy_diff = final_training_accuracy - test_accuracy
    assert accuracy_diff <= 0.05, (
        f"Test accuracy {test_accuracy:.4f} is more than 5% lower than final "
        f"training accuracy {final_training_accuracy:.4f} (difference: {accuracy_diff:.4f})"
    )
'''


def patch_test(data):
    source = data.decode()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name == 'test_model_accuracy_verification']
    if len(nodes) != 1:
        raise ValueError('Expected exactly one original accuracy test')
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    result = ''.join(lines[:node.lineno - 1]) + REPLACEMENT + ''.join(lines[node.end_lineno:])
    before = [ast.dump(n, include_attributes=False) for n in ast.parse(source).body
              if not isinstance(n, ast.FunctionDef) or n.name != node.name]
    after = [ast.dump(n, include_attributes=False) for n in ast.parse(result).body
             if not isinstance(n, ast.FunctionDef) or n.name != node.name]
    assert before == after, 'Other original tests changed'
    return result.encode()


def patch_script(data):
    if data.count(b'-w pytest==8.4.1') != 1 or data.count(b'if [ $? -eq 0 ]; then') != 1:
        raise ValueError('Unexpected verifier dependency/reward bootstrap')
    data = data.replace(b'#!/bin/bash', b'#!/bin/bash\n# Fresh grading invocation: never reuse an old verdict.\nrm -f /logs/verifier/grading-pending.json /logs/verifier/caffe-grader-asset-infrastructure.json /logs/verifier/reward.txt /logs/verifier/reward.json', 1)
    data = data.replace(b'-w pytest==8.4.1', b'-w protobuf==6.33.5 -w lmdb==1.7.5 -w pytest==8.4.1', 1)
    data = data.replace(b'if [ $? -eq 0 ]; then', b'pytest_status=$?\nif [ -e /logs/verifier/grading-pending.json ] || [ -e /logs/verifier/caffe-grader-asset-infrastructure.json ]; then\n  echo "Caffe evaluator adapter requires review; no numeric reward written."\n  exit 0\nfi\nif [ "$pytest_status" -eq 0 ]; then', 1)
    return data


def patch_config(data):
    text = data.decode()
    old = tomllib.loads(text)
    if old['artifacts'] != [] or text.count('artifacts = []') != 1:
        raise ValueError('Unexpected original artifact configuration')
    text = text.replace('artifacts = []', 'artifacts = ' + json.dumps(ARTIFACTS).replace(': ', ' = '), 1)
    new = tomllib.loads(text)
    assert new.pop('artifacts') == ARTIFACTS
    assert new == {k: v for k, v in old.items() if k != 'artifacts'}
    return text.encode()


def build(source, assets, destination):
    source, assets, destination = Path(source), Path(assets), Path(destination)
    if grader.sha256(source) != ORIGINAL_BUNDLE_SHA256:
        raise ValueError('Original bundle pin mismatch')
    # Complete verification here means a tampered/stand-in dataset cannot enter
    # the trusted test tree even if supplied under the official archive name.
    count = sum(1 for _ in grader.official_records(assets / 'test_batch.bin'))
    grader.message_types(assets / 'caffe.desc')
    additions = {
        TASK + 'tests/caffe_official_grader.py': Path(grader.__file__).read_bytes(),
        TASK + 'tests/official-assets/caffe.desc': (assets / 'caffe.desc').read_bytes(),
        TASK + 'tests/official-assets/test_batch.bin': (assets / 'test_batch.bin').read_bytes(),
    }
    changes, original, unchanged_tasks = [], {}, set()
    with tarfile.open(source) as src, destination.open('xb') as stream, tarfile.open(fileobj=stream, mode='w') as dst:
        for member in src.getmembers():
            if member.name in additions:
                raise ValueError('Overlay already exists')
            original[member.name] = member
            data = src.extractfile(member).read() if member.isfile() else None
            if member.name == TASK + 'tests/test_outputs.py':
                data = patch_test(data)
                changes.append(member.name)
            elif member.name == TASK + 'tests/test.sh':
                data = patch_script(data)
                changes.append(member.name)
            elif member.name == TASK + 'task.toml':
                data = patch_config(data)
                changes.append(member.name)
            elif member.name.startswith('tasks/') and not member.name.startswith(TASK):
                parts = member.name.split('/')
                if len(parts) > 2:
                    unchanged_tasks.add(parts[1])
            info = copy.copy(member)
            if data is not None:
                info.size = len(data)
            dst.addfile(info, io.BytesIO(data) if data is not None else None)
        for name, data in sorted(additions.items()):
            info = tarfile.TarInfo(name)
            info.mode, info.size = 0o600, len(data)
            dst.addfile(info, io.BytesIO(data))
    # Independent member bytes/type/mode/link comparison, including all 88
    # unrelated tasks and Caffe instruction/environment/public solution files.
    verified = 0
    with tarfile.open(source) as src, tarfile.open(destination) as dst:
        for name, member in original.items():
            if name in changes:
                continue
            target = dst.getmember(name)
            if (member.type, member.mode, member.linkname) != (target.type, target.mode, target.linkname):
                raise AssertionError(f'Member metadata changed: {name}')
            if member.isfile():
                if src.extractfile(member).read() != dst.extractfile(target).read():
                    raise AssertionError(f'Member bytes changed: {name}')
                verified += 1
    manifest = {'schema_version': 1, 'version': 'caffe-official-test-v4',
        'admission': 'BLOCKED pending non-regenerating adapter-infrastructure handling and root review',
        'source_bundle': str(source), 'source_sha256': grader.sha256(source),
        'bundle': str(destination), 'bundle_sha256': grader.sha256(destination),
        'official_archive_sha256': grader.ARCHIVE_SHA256, 'official_test_sha256': grader.TEST_SHA256,
        'descriptor_sha256': grader.DESCRIPTOR_SHA256,
        'official_test_records': count, 'unchanged_other_task_count': len(unchanged_tasks),
        'unchanged_original_files': verified, 'changed_members': changes,
        'added_members': {k: hashlib.sha256(v).hexdigest() for k, v in additions.items()},
        'agent_instruction_environment_unchanged': True, 'asset_delivery_phase': 'verifier.verify tests upload, after agent',
        'replay_note': 'artifacts plus verifier official-cifar10/replay-means and pinned original image/runtime cache; missing artifact statuses must be checked'}
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def narrow_artifacts(source, destination):
    """Prepare a NEW future overlay; change only our evaluator artifact metadata.

    Does not acquire/expose assets, alter grading, adopt a bundle, or run a model.
    Dynamic runtime/mean/input-domain replay dependencies remain incomplete.
    """
    source, destination = Path(source), Path(destination)
    changed = TASK + 'task.toml'
    with tarfile.open(source) as original:
        raw = original.extractfile(changed).read()
        before = tomllib.loads(raw.decode())
        if before['artifacts'] != ARTIFACTS:
            raise ValueError('Expected our existing broad evaluator artifact selectors')
        line = 'artifacts = ' + json.dumps(ARTIFACTS).replace(': ', ' = ')
        if raw.decode().count(line) != 1:
            raise ValueError('Unexpected artifact metadata formatting')
        replacement = ('artifacts = ' + json.dumps(FIXED_ARTIFACTS).replace(': ', ' = ')).encode()
        updated = raw.replace(line.encode(), replacement, 1)
        after = tomllib.loads(updated.decode())
        assert after.pop('artifacts') == FIXED_ARTIFACTS
        assert after == {k: v for k, v in before.items() if k != 'artifacts'}
        with destination.open('xb') as stream, tarfile.open(fileobj=stream, mode='w') as output:
            for member in original.getmembers():
                data = original.extractfile(member).read() if member.isfile() else None
                info = copy.copy(member)
                if member.name == changed:
                    data, info.size = updated, len(updated)
                output.addfile(info, io.BytesIO(data) if data is not None else None)
    verified = 0
    with tarfile.open(source) as original, tarfile.open(destination) as output:
        assert original.getnames() == output.getnames()
        assert output.extractfile(changed).read() == updated
        for member in original.getmembers():
            other = output.getmember(member.name)
            assert (member.type, member.mode, member.linkname) == (other.type, other.mode, other.linkname)
            if member.isfile() and member.name != changed:
                assert original.extractfile(member).read() == output.extractfile(other).read()
                verified += 1
    manifest = {'schema_version': 1, 'state': 'FUTURE_PROPOSAL_NOT_ADOPTED',
                'source': str(source), 'source_sha256': grader.sha256(source),
                'bundle': str(destination), 'bundle_sha256': grader.sha256(destination),
                'changed_members': [changed], 'changed_fields': ['artifacts'],
                'fixed_artifacts': FIXED_ARTIFACTS, 'unchanged_file_count': verified,
                'instruction_tests_official_assets_other_tasks_byte_identical': True,
                'replay_complete': False,
                'replay_limit': 'Fixed submissions only; dynamic runtime, external means '
                                'and input representation dependencies remain incomplete.'}
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    if sys.argv[1:2] == ['--narrow-artifacts']:
        print(json.dumps(narrow_artifacts(*sys.argv[2:]), indent=2))
    else:
        print(json.dumps(build(*sys.argv[1:]), indent=2))
