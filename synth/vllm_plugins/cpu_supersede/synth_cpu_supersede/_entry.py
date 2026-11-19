"""vllm.general_plugins entry point of synth-cpu-supersede: picks the code path for the installed vLLM.

vLLM < 0.34 runs v2 (`synth_cpu_supersede/__init__.py`, unchanged since v2); vLLM >= 0.34 runs v3 (`vllm034.py`).
The version is read from the imported `vllm` module, not from package metadata (a source tree on PYTHONPATH can
shadow an installed vLLM of another version).
"""
import os
import re


def vllm_version():
    """(major, minor) of the imported vLLM, or None if its version string does not parse."""
    import vllm
    match = re.match(r'(\d+)\.(\d+)', getattr(vllm, '__version__', '') or '')
    return (int(match[1]), int(match[2])) if match else None


def register():
    if os.environ.get('SYNTH_CPU_SUPERSEDE') != '1':
        return
    version = vllm_version()
    if version is None:
        import logging
        logging.getLogger('vllm.synth_cpu_supersede').warning(
            'synth-cpu-supersede: unparsable vLLM version; nothing patched')
        return
    if version >= (0, 34):
        from synth_cpu_supersede import vllm034 as impl
    else:
        import synth_cpu_supersede as impl
    impl.register()
