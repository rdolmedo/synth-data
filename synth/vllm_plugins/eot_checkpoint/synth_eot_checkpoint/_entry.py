"""vllm.general_plugins entry point of synth-eot-checkpoint: picks the code path for the installed vLLM.

vLLM < 0.34 runs v4 (`synth_eot_checkpoint/__init__.py`, unchanged since v4); vLLM 0.34 runs v5 (`vllm034.py`) only
on a build it was validated on (VALIDATED_V5): the plugin caches engine state, so a later build that keeps the names
but changes the Mamba slot layout or the finish-hook order would cache wrong end states silently. Any other >= 0.34
build logs a warning and patches nothing (serving is then stock vLLM). The version is read from the imported `vllm`
module, not from package metadata (a source tree on PYTHONPATH can shadow an installed vLLM of another version).
"""
import os
import re

VALIDATED_V5 = ('0.34.0rc1.dev815+g8f3c672dc',)   # vLLM main 8f3c672dc (#478); add a build after its eot_logprob_check


def vllm_version():
    """(major, minor) of the imported vLLM, or None if its version string does not parse."""
    import vllm
    match = re.match(r'(\d+)\.(\d+)', getattr(vllm, '__version__', '') or '')
    return (int(match[1]), int(match[2])) if match else None


def register():
    if os.environ.get('SYNTH_EOT_CHECKPOINT') != '1':
        return
    version = vllm_version()
    if version is None:
        import logging
        logging.getLogger('vllm.synth_eot_checkpoint').warning(
            'synth-eot-checkpoint: unparsable vLLM version; nothing patched')
        return
    if version >= (0, 34):
        import vllm
        if not any(vllm.__version__ == v or vllm.__version__.startswith(v + '.') for v in VALIDATED_V5):
            import logging
            logging.getLogger('vllm.synth_eot_checkpoint').warning(
                'synth-eot-checkpoint: vLLM %s is not a validated v5 build (%s); nothing patched', vllm.__version__,
                ', '.join(VALIDATED_V5))
            return
        from synth_eot_checkpoint import vllm034 as impl
    else:
        import synth_eot_checkpoint as impl
    impl.register()
