"""GPU discovery and preflight for a serving allocation.

HTCondor hands out abbreviated `GPU-xxxx` UUIDs; CUDA accepts them, NVML (used
by vLLM) needs full UUIDs. nvidia-smi can look healthy while cuInit fails
(cpu403), so every assigned device must also retain a real CUDA context.
"""
import ctypes
import json
import os
import subprocess
import sys


def assigned():
    raw = os.environ.get('_CONDOR_AssignedGPUs') or os.environ.get('CUDA_VISIBLE_DEVICES', '')
    return [x.strip() for x in raw.split(',') if x.strip()]


def expand(ids, timeout=60):
    """Expand UUID prefixes to the unique full UUIDs reported by nvidia-smi."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=timeout, check=True).stdout
    available = [line.split(',')[1].strip() for line in out.splitlines() if ',' in line]
    full = []
    for item in ids:
        if not item.startswith('GPU-'):
            full.append(item)
            continue
        matches = [u for u in available if u.startswith(item)]
        if len(matches) != 1:
            raise RuntimeError(f'Cannot uniquely expand assigned GPU {item}: {matches}')
        full.append(matches[0])
    if len(set(full)) != len(full):
        raise RuntimeError(f'Duplicate assigned GPU identifiers: {full}')
    return full


def indices(gpus, timeout=60):
    """nvidia-smi indices of assigned GPUs (full UUIDs from expand(), or indices already), in the order given. For
    engines that parse CUDA_VISIBLE_DEVICES as integers: vLLM 0.20.1 (platforms.interface.device_id_to_physical_
    device_id) fails on UUIDs. Inside an HTCondor job nvidia-smi lists only the job's GPUs, numbered from 0."""
    if all(g.isdigit() for g in gpus):
        return list(gpus)
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=timeout, check=True).stdout
    by_uuid = {u.strip(): i.strip() for i, u in (line.split(',')[:2] for line in out.splitlines() if ',' in line)}
    missing = [g for g in gpus if not g.isdigit() and g not in by_uuid]
    if missing:
        raise RuntimeError(f'No nvidia-smi index for assigned GPUs {missing}: {by_uuid}')
    return [g if g.isdigit() else by_uuid[g] for g in gpus]


def cuda_contexts():
    """Run in a child: cuInit + primary-context retain on every visible device."""
    report = {'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'calls': []}
    cuda = ctypes.CDLL('libcuda.so.1')
    def call(name, *args):
        code = getattr(cuda, name)(*args)
        label = ctypes.c_char_p()
        cuda.cuGetErrorName(code, ctypes.byref(label))
        report['calls'].append({'function': name, 'code': code,
                                'name': label.value.decode() if label.value else None})
        return code == 0
    if call('cuInit', 0):
        count = ctypes.c_int()
        if call('cuDeviceGetCount', ctypes.byref(count)):
            report['device_count'] = count.value
            for index in range(count.value):
                context = ctypes.c_void_p()
                if call('cuDevicePrimaryCtxRetain', ctypes.byref(context), index):
                    call('cuDevicePrimaryCtxRelease', index)
    return report


def preflight(gpus, timeout=120, max_used_mib=2048):
    """Bounded nvidia-smi + CUDA context check; returns (passed, report). An assigned GPU that already holds more than
    max_used_mib (another process on it) fails too: vLLM would exit on every start for lack of free memory."""
    report = {'assigned': gpus}
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version,memory.used',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=timeout)
        report['nvidia_smi'] = {'returncode': smi.returncode, 'stdout': smi.stdout, 'stderr': smi.stderr}
        child = subprocess.run([sys.executable, '-c', 'import json,synth.gpu as g; print(json.dumps(g.cuda_contexts()))'],
                               capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(gpus),
                                    'PYTHONPATH': os.environ.get('PYTHONPATH', '')})
        report['cuda'] = json.loads(child.stdout) if child.returncode == 0 and child.stdout.strip() else {
            'returncode': child.returncode, 'stderr': child.stderr[-4000:]}
    except subprocess.TimeoutExpired as exc:
        report['timeout'] = repr(exc)
        return False, report
    cuda = report.get('cuda', {})
    for line in smi.stdout.splitlines():
        fields = [f.strip() for f in line.split(',')]
        used = fields[-1].split()[0] if len(fields) >= 6 else ''
        if (fields[0] in gpus or fields[1] in gpus) and used.isdigit() and int(used) > max_used_mib:
            report.setdefault('busy', {})[fields[1]] = int(used)
    passed = (report['nvidia_smi']['returncode'] == 0 and cuda.get('device_count') == len(gpus)
              and all(c['code'] == 0 for c in cuda.get('calls', [])) and not report.get('busy'))
    return passed, report
