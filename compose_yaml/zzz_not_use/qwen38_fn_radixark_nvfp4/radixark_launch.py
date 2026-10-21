"""Run the cached RadixArk checkpoint with native NVFP4 and indexed MTP reads."""
import json
import os
from pathlib import Path
import runpy
import sys


def mtp_files(folder, files, config):
    if (config is None or not getattr(config, 'is_draft_model', False)
            or getattr(config.hf_config, 'model_type', None) != 'qwen4_exp'):
        return files
    index_path = Path(folder) / 'model.safetensors.index.json'
    if not index_path.is_file():
        return files
    index = json.loads(index_path.read_text())['weight_map']
    needed = {str((Path(folder) / name).resolve()) for key, name in index.items() if key.startswith('mtp.')}
    if not needed:
        return files
    selected = [name for name in files if str(Path(name).resolve()) in needed]
    if {str(Path(name).resolve()) for name in selected} != needed:
        raise ValueError('RadixArk MTP index references a missing cached shard')
    print(f'RADIXARK_MTP_SHARD_FILTER selected={len(selected)} total={len(files)}', flush=True)
    return selected


if __name__ == '__main__':
    import qad_loader
    qad_loader.mtp_files = mtp_files
    helper = runpy.run_path('/opt/sparktalk-flash-next/launch.py')
    args = helper['server_arguments'](sys.argv[1:], os.getenv('SPARKTALK_FLASH_NEXT_DRAFT_VOCAB', 'ko64k'))
    sys.argv = ['sglang.launch_server', *args]
    runpy.run_module('sglang.launch_server', run_name='__main__')
