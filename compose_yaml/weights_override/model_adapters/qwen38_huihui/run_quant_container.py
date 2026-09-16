import subprocess,pathlib,sys
root=pathlib.Path.home();project=pathlib.Path(__file__).resolve().parents[4];cache=root/'.cache/huggingface'
cmd=['docker','run','--rm','--name','huihui-quant-cpu','--network','none','--read-only','--tmpfs','/tmp:rw,size=1g','--memory','6g','--cpus','2','--user','1000:1000','--entrypoint','python3','-e','PYTHONPATH=/modelopt:/opt:/work','-e','OMP_NUM_THREADS=2']
for a,b,ro in [(project/'compose_yaml/weights_override/weights_core','/weights_core',True),(cache,'/hf','--probe-only' in sys.argv),(root/'.cache/model-download-jobs/qwen38fn-huihui/source-bf16','/source-raw',True),(project/'compose_yaml/weights_override/model_adapters/qwen38_huihui','/work',True),(project/'compose_yaml/weights_override/model_adapters/qwen38_huihui/docs','/reports',True),(project/'compose_yaml/weights_override/vendor/Model-Optimizer','/modelopt',True),(root/'.cache/model-tools/qwen38fn-huihui-venv/lib/python3.12/site-packages/gguf','/opt/gguf',True)]:
 cmd+=['--mount',f'type=bind,src={a},dst={b}'+(',readonly' if ro else '')]
cmd+=['dgx-sglang-qwen38-fn:sm121-b12x-head-v1','/work/patch_experts.py','--cache','/hf','--audit','/reports/audit.json','--source-raw','/source-raw','--output','/hf/Qwen3.8-Flash-Next-Huihui-GGUFDelta-NVFP4']+sys.argv[1:]
subprocess.run(cmd,check=True)
