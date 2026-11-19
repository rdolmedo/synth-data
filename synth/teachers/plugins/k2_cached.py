"""Optional numerically checked MoVA weight-layout optimization, exact same model.

Stock vLLM stacks all immutable BF16 value-expert weights on every forward.
This candidate builds the identical contiguous tensor once, rebinding the expert
Parameter views to that storage. It retains no duplicate copy and changes no
arithmetic, precision, weight value, router, or checkpoint loader mapping.
"""
import json
import os
from pathlib import Path
import time
import torch
from vllm.model_executor.models.k2_horizon import (
    K2HorizonForCausalLM,K2HorizonMoVAAttention,calc_router_weights,fused_mova_impl,
)

class CachedMoVAAttention(K2HorizonMoVAAttention):
    def compute_mova_v_sparse(self, hidden_states):
        router_logits, _ = self.v_router(hidden_states)
        routing_weights, selected_values = calc_router_weights(
            router_logits=router_logits,
            e_score_correction_bias=self.v_router.bias,
            score_func=self.router_score_func,
            top_k=self.num_experts_per_tok,
            scaling_factor=self.router_scaling_factor,
        )
        return fused_mova_impl(
            config=self.fused_mova_config,
            hidden_states=hidden_states.contiguous(),
            w1=self._cached_mova_values,
            topk_weights=routing_weights.contiguous(),
            topk_ids=selected_values.contiguous(),
            global_num_experts=self.num_experts,
            expert_map=None,
        )

class K2HorizonCachedForCausalLM(K2HorizonForCausalLM):
    def load_weights(self,weights):
        loaded=super().load_weights(weights)
        proof=[]
        with torch.no_grad():
            modules=[(name,m) for name,m in self.named_modules() if isinstance(m,K2HorizonMoVAAttention)]
            for number,(name,module) in enumerate(modules):
                stacked=torch.stack([e.weight for e in module.v_experts],dim=0).contiguous()
                if stacked.dtype!=torch.bfloat16: raise RuntimeError('MoVA value experts must remain BF16')
                for i,expert in enumerate(module.v_experts):
                    if not torch.equal(stacked[i],expert.weight): raise RuntimeError('Value-expert layout mismatch')
                    expert.weight.data=stacked[i]
                module.register_buffer('_cached_mova_values',stacked,persistent=False)
                # Compare the original and cached arithmetic on representative
                # hidden-state batches before selecting the optimized class.
                checks=[]
                if number in (0,len(modules)//2,len(modules)-1):
                    generator=torch.Generator(device=stacked.device).manual_seed(20261015+number)
                    for batch in (1,16,128):
                        hidden=torch.randn(batch,stacked.shape[-1],dtype=stacked.dtype,device=stacked.device,generator=generator)
                        original=K2HorizonMoVAAttention.compute_mova_v_sparse(module,hidden)
                        cached=CachedMoVAAttention.compute_mova_v_sparse(module,hidden)
                        equal=torch.equal(original,cached)
                        if not equal: raise RuntimeError(f'Cached MoVA output differs: {name} batch={batch}')
                        checks.append({'batch':batch,'bitwise_equal':equal})
                module.__class__=CachedMoVAAttention
                proof.append({'module':name,'dtype':str(stacked.dtype),'shape':list(stacked.shape),'weights_bitwise_equal':True,'output_checks':checks})
        path=Path(os.environ['K2_LAYOUT_PROOF'])
        path.write_text(json.dumps({'timestamp':time.time(),'candidate':'cached_mova_values','proof':proof},indent=2)+'\n')
        return loaded
