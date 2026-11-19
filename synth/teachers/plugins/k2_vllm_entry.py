"""Register the optional layout candidate in parent and spawned vLLM workers."""
from vllm import ModelRegistry
ModelRegistry.register_model('K2HorizonCachedForCausalLM','synth.teachers.plugins.k2_cached:K2HorizonCachedForCausalLM')
if __name__=='__main__':
    from vllm.entrypoints.cli.main import main
    main()
