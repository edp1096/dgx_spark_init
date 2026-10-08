import json
from types import SimpleNamespace as NS
import pytest
from qad_loader import mtp_files


def test_only_mtp_files_and_untouched_target(tmp_path):
    files=[str(tmp_path/'main.safetensors'),str(tmp_path/'draft.safetensors')]
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{
        'model.embed_tokens.weight':'main.safetensors',
        'mtp.layers.0.weight':'draft.safetensors',
        'mtp.norm.weight':'draft.safetensors'}}))
    config=NS(is_draft_model=True,quantization='modelopt_mixed',hf_config=NS(model_type='qwen4_exp'))
    assert mtp_files(tmp_path,files,config)==files[1:]
    config.is_draft_model=False
    assert mtp_files(tmp_path,files,config) is files
    config.is_draft_model=True
    with pytest.raises(ValueError,match='missing'):
        mtp_files(tmp_path,files[:1],config)
    config.hf_config.model_type='another_model'
    assert mtp_files(tmp_path,files,config) is files
