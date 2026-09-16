"""TP2 launch configuration. Original checkpoint files remain read-only."""
import json,os,sys
from pathlib import Path

MODEL='/hf/hub/models--dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4/snapshots/be794b990578ef3031eccf9f28e675a289a09ee9'
def arguments(env=os.environ):
    rank=int(env['QWEN_TP2_RANK']);context=int(env.get('QWEN_TP2_CONTEXT','262144'))
    if rank not in (0,1) or context not in (262144,524288,1048576):raise ValueError('Unsupported TP2 rank/context')
    if env.get('SPARKTALK_FLASH_NEXT_DRAFT_VOCAB','off')!='off':raise ValueError('TP2 shortlist is not qualified')
    args=['--model-path',MODEL,'--served-model-name',env.get('QWEN_TP2_MODEL','qwen3.8-flash-next'),
          '--host',env.get('QWEN_TP2_BIND','0.0.0.0'),'--port',env.get('QWEN_TP2_API_PORT','8012'),
          '--moe-runner-backend','flashinfer_cutlass','--tp-size','2','--nnodes','2','--node-rank',str(rank),
          '--dist-init-addr',env.get('QWEN_TP2_HEAD','10.200.0.1')+':'+env.get('QWEN_TP2_DIST_PORT','29780'),
          '--load-format','auto','--weight-loader-drop-cache-after-load',
          '--context-length',str(context),'--max-total-tokens',str(context+65536),
          '--mem-fraction-static','0.70','--chunked-prefill-size','1024',
          '--max-running-requests','1','--page-size','64','--cuda-graph-max-bs-decode','1',
          '--kv-cache-dtype','bfloat16','--prefill-attention-backend','triton',
          '--decode-attention-backend','trtllm_mha','--quantization','modelopt_fp4',
          '--fp4-gemm-backend','flashinfer_cutlass','--ple-offload-embedding',
          '--ple-offload-backend','file','--ple-offload-dir','/ple',
          '--mamba-radix-cache-strategy','extra_buffer','--reasoning-parser','qwen3',
          '--tool-call-parser','qwen3_coder','--mm-feature-transport','cpu',
          '--speculative-algorithm','NEXTN','--speculative-num-steps','3',
          '--speculative-eagle-topk','1','--speculative-num-draft-tokens','4',
          '--speculative-draft-model-quantization','unquant','--disable-flashinfer-autotune',
          '--enable-metrics']
    # No automatic truncation: long-context validation must reject oversized inputs.
    if context>262144:
        cfg=json.loads(Path(MODEL,'config.json').read_text());text=cfg.get('text_config',cfg)
        rope=dict(text['rope_parameters'])
        rope.update(rope_type='yarn',factor=context/262144,original_max_position_embeddings=262144)
        override={'max_position_embeddings':context,'rope_parameters':rope,'rope_scaling':rope}
        if 'text_config' in cfg:override['text_config']=dict(override)
        args+=['--json-model-override-args',json.dumps(override)]
    return args

if __name__=='__main__':
    # Isolated container-local adapter update; keep immutable base image intact.
    Path('/usr/local/lib/python3.12/dist-packages/sglang_b12x_probe.py').write_bytes(Path(__file__).with_name('adapter.py').read_bytes())
    if os.environ.get('SGLANG_B12X_PROBE_HEAD')=='1':
        p=Path('/sgl-workspace/sglang/python/sglang/srt/layers/logits_processor.py')
        source=p.read_text()
        anchor='        elif hasattr(lm_head, "weight"):\n            # Normal linear layer'
        assert source.count(anchor)==1
        replacement='        elif hasattr(lm_head, "weight"):\n            # Narrow opt-in: preserve BF16 weights/logits and all outer TP logic.\n            if (not self.use_fp32_lm_head and self.rl_on_policy_target is None\n                and hidden_states.dtype == torch.bfloat16\n                and lm_head.weight.dtype == torch.bfloat16\n                and hidden_states.shape == (1, 2560)\n                and lm_head.weight.shape == (124160, 2560)):\n                from b12x.gemm.bf16_gemv import mm\n                if not getattr(self, "_b12x_probe_head_logged", False):\n                    print("B12X_PROBE_HEAD active: BF16 1x2560 by 124160x2560", flush=True)\n                    self._b12x_probe_head_logged = True\n                return mm(hidden_states, lm_head.weight, output_dtype=torch.bfloat16)\n            # Normal linear layer'
        p.write_text(source.replace(anchor,replacement))
    args=arguments()
    print('QWEN_TP2_ARGS '+json.dumps(args),flush=True)
    os.execv(sys.executable,[sys.executable,'-m','sglang.launch_server',*args])
