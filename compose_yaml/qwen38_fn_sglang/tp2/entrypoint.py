"""TP2 launch configuration. Original checkpoint files remain read-only."""
import json,os,sys
from pathlib import Path

MODEL='/hf/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'
def arguments(env=os.environ):
    rank=int(env['QWEN_TP2_RANK']);context=int(env.get('QWEN_TP2_CONTEXT','262144'))
    if rank not in (0,1) or context not in (262144,524288,1048576):raise ValueError('Unsupported TP2 rank/context')
    mode=env.get('SPARKTALK_FLASH_NEXT_DRAFT_VOCAB','ko64k')
    if mode=='ko64k' and not Path('/sgl-workspace/sglang/python/sglang/srt/speculative/tp_vocab.py').is_file():
        raise ValueError('TP2 ko64k requires the TP2 shortlist image; use SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=off with older images')
    chunk=int(env.get('QWEN_TP2_CHUNK','1024'))
    if chunk not in (1024,2048,4096,8192):raise ValueError('Unsupported chunk')
    args=['--model-path',MODEL,'--served-model-name',env.get('QWEN_TP2_MODEL','edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4'),
          '--host',env.get('QWEN_TP2_BIND','0.0.0.0'),'--port',env.get('QWEN_TP2_API_PORT','8012'),
          '--tp-size','2','--nnodes','2','--node-rank',str(rank),
          '--dist-init-addr',env.get('QWEN_TP2_HEAD','10.200.0.1')+':'+env.get('QWEN_TP2_DIST_PORT','29780'),
          '--load-format','auto','--weight-loader-drop-cache-after-load',
          '--context-length',str(context),'--max-total-tokens',str(context+65536),
          '--mem-fraction-static','0.70','--chunked-prefill-size',str(chunk),
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
    sys.path.insert(0,'/opt/sparktalk-flash-next')
    from launch import server_arguments
    return server_arguments(args,mode,Path('/opt/sparktalk-flash-next'))

if __name__=='__main__':
    args=arguments()
    print('QWEN_TP2_ARGS '+json.dumps(args),flush=True)
    os.execv(sys.executable,[sys.executable,'-m','sglang.launch_server',*args])
