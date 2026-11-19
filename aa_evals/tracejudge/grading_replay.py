"""Read-only validation of the one reviewed reshard native grading replay.

Administrative host certificates never come from the agent archive. This module
has no generation, network, submission, or native-execution entrypoint.
"""
from pathlib import Path
import hashlib
import json

KIND = 'reshard-a1-native-grading-replay-v4'
KEY = 'reshard-c4-data__a1__r0'
APPROVED = {
    'receipt': '1aa09c558f5e50be52a57308ae1225521f984b9548c546a56d8b071cbf5e2bdb',
    'plan': '9b5192b54691504a4aacf3e6571da2558b1b3362e124a3716b1976bae9c357b2',
    'runner': 'a7d1747ca3213ef97a7402f4ab151d585179c26e1c82b8ce74a5c1814ae8f453',
    'source_config': '098fa1bf48de7113ed000d6bf4114c97cde768bc74df30df79a45dee4b3023c2',
}

def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('grading replay proof must be a regular file')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate grading replay field')
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def proof(spec):
    if not isinstance(spec, dict) or set(spec) != {'path', 'sha256'} or not Path(spec['path']).is_absolute():
        raise ValueError('absolute hash-bound grading replay proof required')
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('grading replay proof SHA256 mismatch')
    return Path(spec['path'])


def validate(run_dir, key, certificate, outcome):
    """Raise on any unsupported/incomplete certificate; return current evidence only."""
    run_dir = Path(run_dir)
    if key != KEY:
        raise ValueError('unsupported grading replay physical key')
    certificate = Path(certificate)
    certificate_sha = sha(certificate)
    cert = read(certificate)
    if (set(cert) != {'schema_version', 'kind', 'key', 'verdict', 'source', 'plan', 'receipt', 'root_audit'}
            or type(cert['schema_version']) is not int or cert['schema_version'] != 1
            or cert['kind'] != KIND or cert['key'] != key or cert['verdict'] != 'adopt_native_grading'):
        raise ValueError('unsupported grading replay certificate')
    audit = read(proof(cert['root_audit']))
    if (type(audit.get('schema_version')) is not int or audit['schema_version'] != 1
            or audit.get('state') != 'ROOT_REVIEWED_SCORING_ONLY_ADOPTION'
            or audit.get('kind') != KIND or audit.get('key') != key or audit.get('verdict') != cert['verdict']
            or audit.get('certificate_payload_sha256') != digest({k: v for k, v in cert.items() if k != 'root_audit'})
            or audit.get('original_failure') != 'grader_fixture_before_test_call'
            or audit.get('generation_backend_failure') is not False
            or audit.get('originals_unchanged') is not True
            or not isinstance(audit.get('evidence'), str) or not audit['evidence'].strip()):
        raise ValueError('root grading replay audit is incomplete or mismatched')
    plan_path, receipt_path = proof(cert['plan']), proof(cert['receipt'])
    if cert['plan']['sha256'] != APPROVED['plan'] or cert['receipt']['sha256'] != APPROVED['receipt']:
        raise ValueError('unapproved grading replay plan or receipt')
    plan, receipt = read(plan_path), read(receipt_path)
    if (plan.get('key') != key or type(plan.get('schema_version')) is not int or plan.get('schema_version') != 4
            or receipt.get('key') != key or receipt.get('state') != 'NATIVE_GRADING_COMPLETE_PENDING_ROOT_ADOPTION'
            or receipt.get('plan_file_sha256') != APPROVED['plan']
            or plan.get('runner_sha256') != APPROVED['runner'] or receipt.get('runner_sha256') != APPROVED['runner']):
        raise ValueError('unsupported native grading replay provenance')
    for field in ('generation_calls', 'model_clients_constructed', 'agent_constructions'):
        if type(receipt.get(field)) is not int or receipt[field] != 0:
            raise ValueError('grading replay constructed agent/model or generated content')
    if (receipt.get('source_run_writes') is not False or receipt.get('originals_unchanged') is not True
            or receipt.get('grader_data_before_after_unchanged') is not True
            or type(plan.get('generation_calls')) is not int or plan.get('generation_calls') != 0 or plan.get('original_writes') is not False
            or plan.get('passing_control_regrades') is not False):
        raise ValueError('grading replay changed protected sources or scope')
    sources = cert['source']
    if not isinstance(sources, dict) or set(sources) != {'config', 'archive', 'attempt', 'outcome'}:
        raise ValueError('grading replay source bindings incomplete')
    if sources['config'].get('sha256') != APPROVED['source_config']:
        raise ValueError('unapproved original generation config')
    paths = {name: proof(spec) for name, spec in sources.items()}
    source_dir = paths['archive'].parent
    if (source_dir.name != key or paths['archive'].name != 'trial.tar.gz'
            or paths['attempt'] != source_dir / 'attempt.json' or paths['outcome'] != source_dir / 'outcome.json'
            or paths['config'] != source_dir.parent.parent / 'config.json'):
        raise ValueError('grading replay source paths do not identify one native attempt')
    for name in ('archive', 'attempt', 'outcome'):
        if plan.get('protected_originals', {}).get(str(paths[name])) != sources[name]['sha256']:
            raise ValueError('grading replay plan does not protect original attempt')
    raw = read(paths['outcome'])
    if (raw.get('infrastructure_failure') is not False or raw.get('exception_type') is not None
            or raw.get('exception_message') is not None or raw.get('score') != 0
            or (raw.get('stats') or {}).get('end_state') != 'finished'):
        raise ValueError('grading replay cannot override primary generation/backend infrastructure')
    current = run_dir / 'attempts' / key
    for name, filename in (('archive', 'trial.tar.gz'), ('attempt', 'attempt.json')):
        if sha(current / filename) != sources[name]['sha256']:
            raise ValueError('grading replay does not bind current immutable generation')
    if sha(current / 'outcome.json') != sources['outcome']['sha256']:
        provenance = outcome.get('grading_replay') or {}
        if (outcome.get('infrastructure_failure') is not False
                or provenance.get('certificate_sha256') != certificate_sha
                or provenance.get('original_outcome_sha256') != sources['outcome']['sha256']):
            raise ValueError('current derived outcome lacks original grading replay binding')
    restored = {name: item['sha256'] for name, item in plan['restored_files'].items()}
    if not restored or receipt.get('restored_before') != restored or receipt.get('restored_after') != restored:
        raise ValueError('saved agent files changed during grading replay')
    for phase in ('before', 'after'):
        call = receipt.get('original_call_' + phase) or {}
        for field in ('original_loader_arguments', 'same_cache_files', 'same_fingerprint', 'packages_unchanged', 'no_arrow_rebuild', 'no_payload_redownload'):
            if call.get(field) is not True:
                raise ValueError('original grader fixture call proof incomplete')
        data = receipt['dataset']
        if (call.get('rows') != data.get('rows') or call.get('fingerprint') != data.get('fingerprint')
                or call.get('cache_files') != data.get('cache_files') or call.get('source_asset_sha256') != data.get('asset_sha256')
                or len(call.get('official_heads', [])) != 2
                or any(head != {'revision': data['revision'], 'etag': data['asset_sha256'], 'bytes': data['asset_bytes']} for head in call['official_heads'])):
            raise ValueError('official grader data provenance differs')
    if receipt.get('grader_network') != {'HF_HUB_OFFLINE': '0', 'HF_DATASETS_OFFLINE': '0'}:
        raise ValueError('unexpected evaluator networking policy')
    trial = receipt_path.parent / 'trial' / 'verifier'
    if sha(trial / 'ctrf.json') != receipt.get('ctrf_sha256') or sha(trial / 'test-stdout.txt') != receipt.get('verifier_log_sha256'):
        raise ValueError('native verifier proof bytes changed')
    ctrf = read(trial / 'ctrf.json')['results']
    tests, summary = ctrf['tests'], ctrf['summary']
    if (len(tests) != 1 or tests[0].get('name') != 'test_outputs.py::test_compress_decompress_workflow'
            or tests[0].get('status') not in ('passed', 'failed')
            or summary.get('tests') != 1 or any(summary.get(k) != 0 for k in ('skipped', 'pending', 'other'))
            or summary.get('passed') != int(tests[0]['status'] == 'passed')
            or summary.get('failed') != int(tests[0]['status'] == 'failed')):
        raise ValueError('native verifier did not execute the original test call')
    reward = receipt['verifier_result']['rewards']['reward']
    reward_file = trial / 'reward.txt'
    if (type(reward) not in (int, float) or reward not in (0, 1)
            or reward != int(tests[0]['status'] == 'passed')
            or sha(reward_file) != receipt['reward_file']['sha256']
            or reward_file.read_text().strip() != str(int(reward))):
        raise ValueError('native verifier binary reward mismatch')
    if outcome.get('grading_replay') and (outcome.get('score') != reward or outcome.get('reward') != reward):
        raise ValueError('derived score differs from validated native replay')
    return {'kind': KIND, 'key': key, 'certificate_sha256': certificate_sha,
            'root_audit_sha256': cert['root_audit']['sha256'], 'receipt_sha256': cert['receipt']['sha256'],
            'plan_sha256': cert['plan']['sha256'], 'source_config_sha256': sources['config']['sha256'],
            'original_outcome_sha256': sources['outcome']['sha256'], 'score': reward,
            'historical_failure': 'original grader fixture failed before the test call; preserved unchanged',
            'current_grading': 'original native verifier executed saved files; evaluator networking permitted; no generation'}


def trace_evidence(run_dir, key, outcome):
    path = Path(run_dir) / 'attempts' / key / 'grading-replay-review.json'
    if not path.exists() and not path.is_symlink() and 'grading_replay' not in outcome:
        return {}
    try:
        evidence = validate(run_dir, key, path, outcome)
        return {'grading_replay_review_pending': False, 'grading_replay': evidence}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {'grading_replay_review_pending': True, 'grading_replay_review_error': str(exc)[:400]}
