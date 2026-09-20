"""Restrict Qwen4 MTP reads to files actually containing its branch."""
import json
from pathlib import Path


def mtp_files(folder, files, config):
    if (config is None or not getattr(config, 'is_draft_model', False)
            or getattr(config.hf_config, 'model_type', None) != 'qwen4_exp'
            or getattr(config, 'quantization', None) != 'modelopt_mixed'):
        return files
    index_path = Path(folder) / 'model.safetensors.index.json'
    if not index_path.is_file():
        return files
    index = json.loads(index_path.read_text())['weight_map']
    needed = {str((Path(folder)/file).resolve()) for name,file in index.items()
              if name.startswith('mtp.')}
    if not needed:
        return files
    selected = [file for file in files if str(Path(file).resolve()) in needed]
    if {str(Path(file).resolve()) for file in selected} != needed:
        raise ValueError('MTP index references a missing checkpoint shard')
    print(f'QAD_MTP_SHARD_FILTER selected={len(selected)} total={len(files)}',flush=True)
    return selected
