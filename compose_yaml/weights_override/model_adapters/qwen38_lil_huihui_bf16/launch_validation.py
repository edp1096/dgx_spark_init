"""Launch an isolated, explicit local model qualification on the idle worker."""
import argparse
import json
from pathlib import Path
import subprocess

IMAGE = 'dgx-sglang-qwen38-qad:sm121-v5-memory'
IMAGE_ID = 'sha256:95b4633b7224deaad847a553c6550891fd3bdad97ad9d8e0c70c843239d26e72'


def command(model, cache, name, port, context):
    if context not in [65536,1048576]:raise ValueError('Unreviewed context size')
    model,cache=model.resolve(),cache.resolve()
    if not (model/'config.json').is_file() or not (model/'model.safetensors.index.json').is_file():
        raise ValueError('Incomplete local model')
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    if any(not (model/f).is_file() for f in set(index.values())):raise ValueError('Missing weight shard')
    ple=[n for n in index if 'ngram_embedding.shard_' in n and n.endswith('.weight_scale')]
    if len(ple)!=128:raise ValueError('Expected the packed 128-shard QAD PLE')
    # Upstream hybrid export dropped this field while retaining packed PLE.
    # Keep source files unchanged and make the loader choice explicit at launch.
    override={'text_config':{'ple_embedding_dtype':'nvfp4'}}
    if context==1048576:
        override['text_config']['rope_parameters']={'mrope_interleaved':True,'mrope_section':[11,11,10],
            'rope_type':'yarn','rope_theta':10000000,'partial_rotary_factor':.25,
            'factor':4.,'original_max_position_embeddings':262144}
    env={'SGLANG_QAD_B12X_GDN':'1','SGLANG_QAD_PLE_IO_URING':'1','B12X_POLICY_MODE':'auto',
        'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
        'B12X_COMPILE_CACHE_DIR':'/cache/b12x-compile','TRITON_CACHE_DIR':'/cache/triton',
        'TORCHINDUCTOR_CACHE_DIR':'/cache/inductor','MAX_JOBS':'1','TORCHINDUCTOR_COMPILE_THREADS':'4',
        'SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB':'0.5','SPARKTALK_FLASH_NEXT_DRAFT_VOCAB':'ko64k',
        'SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN':'1'}
    cmd=['docker','run','-d','--name',name,'--restart','no','--gpus','all','--ipc','host',
         '--memory','112g','--memory-swap','112g','--cpuset-cpus','5-9,15-19',
         '--security-opt','seccomp=unconfined','--ulimit','memlock=-1:-1',
         '-p',f'{port}:30000','-v',f'{model}:/model:ro','-v',f'{cache}:/cache',
         '--entrypoint','python3']
    for k,v in env.items():cmd+=['-e',f'{k}={v}']
    cmd += [IMAGE,'/opt/sparktalk-flash-next/launch.py',
        '--model-path','/model','--served-model-name',name,'--host','0.0.0.0','--port','30000',
        '--load-format','auto','--weight-loader-drop-cache-after-load','--tp-size','1',
        '--context-length',str(context),'--mem-fraction-static','.86' if context==1048576 else '.80',
        '--chunked-prefill-size','4096','--max-total-tokens',str(context),'--max-running-requests','1',
        '--page-size','64','--cuda-graph-max-bs-decode','1',
        '--prefill-attention-backend','triton','--decode-attention-backend','trtllm_mha',
        '--kv-cache-dtype','fp8_e4m3','--quantization','modelopt_mixed',
        '--moe-runner-backend','flashinfer_cutlass','--disable-shared-experts-fusion',
        '--fp4-gemm-backend','flashinfer_cutlass','--ple-offload-embedding',
        '--ple-offload-backend','file','--ple-offload-dir','/cache/ple',
        '--mamba-radix-cache-strategy','extra_buffer','--max-mamba-cache-size','8',
        '--reasoning-parser','qwen3','--tool-call-parser','qwen3_coder','--mm-feature-transport','cpu',
        '--speculative-algorithm','NEXTN','--speculative-num-steps','3',
        '--speculative-eagle-topk','1','--speculative-num-draft-tokens','4',
        '--speculative-draft-model-quantization','modelopt_mixed','--disable-flashinfer-autotune',
        '--enable-metrics','--json-model-override-args',json.dumps(override)]
    return cmd


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--port',type=int,default=30125)
    p.add_argument('--context',type=int,default=65536);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    cmd=command(a.model,a.cache,a.name,a.port,a.context)
    actual=subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
    if actual!=IMAGE_ID:raise ValueError('Unexpected runtime image')
    if a.dry_run:print(json.dumps(cmd,indent=2))
    else:
        a.cache.mkdir(parents=True,exist_ok=True)
        if subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+a.name+'$'],text=True).strip():
            raise ValueError('Validation container already exists; inspect it before relaunching')
        subprocess.run(cmd,check=True)
