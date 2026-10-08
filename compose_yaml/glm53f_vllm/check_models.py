#!/usr/bin/env python3
"""Fast preflight for missing/truncated checkpoint shards before cluster startup."""
import json
import struct
import sys
from pathlib import Path


def check_tensor_file(path):
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError(f'잘린 파일: {path}')
        size=struct.unpack('<Q',raw)[0]
        if not 0<size<=64*1024*1024:raise ValueError(f'잘못된 헤더: {path}')
        header=json.loads(f.read(size))
    tensors=[v for k,v in header.items() if k!='__metadata__']
    if not tensors:raise ValueError(f'빈 가중치: {path}')
    end=max(v['data_offsets'][1] for v in tensors)
    if path.stat().st_size!=8+size+end:raise ValueError(f'불완전한 파일: {path}')


def check(model, draft=None):
    config=json.loads((model/'config.json').read_text())
    if config.get('architectures') != ['Glm5NextForConditionalGeneration']:
        raise ValueError('GLM5Next architecture required')
    quant=config.get('quantization_config', {})
    if quant.get('quant_method') != 'modelopt' or quant.get('quant_algo') != 'NVFP4':
        raise ValueError('ModelOpt NVFP4 checkpoint required')
    index=json.loads((model/'model.safetensors.index.json').read_text())
    weights=index['weight_map']
    names=set(weights.values())
    if not names:raise ValueError('Empty checkpoint index')
    for name in sorted(names):
        if Path(name).name!=name:raise ValueError('Invalid shard path')
        check_tensor_file(model/name)
        with (model/name).open('rb') as f:
            size=struct.unpack('<Q', f.read(8))[0]
            header=json.loads(f.read(size))
        if any(key not in header for key, shard in weights.items() if shard==name):
            raise ValueError('Index tensor missing from shard: '+name)
    for name in ['tokenizer.json','tokenizer_config.json']:
        json.loads((model/name).read_text())
    receipt=model/'transfer-manifest.json'
    if receipt.exists():
        report=json.loads(receipt.read_text())
        if report.get('status')!='candidate_verified':
            raise ValueError('Abliterated candidate verification incomplete')
        if set(report.get('output_shard_hashes',{})) != names:
            raise ValueError('Candidate shard manifest mismatch')
    if draft is not None:
        check_draft(draft)


def check_draft(draft):
    config=json.loads((draft/'config.json').read_text())
    if config.get('architectures') != ['DFlash2DraftModel']:
        raise ValueError('GLM DFlash2 checkpoint required; legacy EXL3 draft is incompatible')
    if (config.get('num_target_layers') != 45
            or config.get('vocab_size') != 154880
            or config.get('dflash_config',{}).get('target_layer_ids') != [5,14,24,33,42]):
        raise ValueError('GLM DFlash2 target configuration mismatch')
    check_tensor_file(draft/'model.safetensors')


def main():
    model=Path(sys.argv[1]);host=sys.argv[2] if len(sys.argv)>2 else '호스트'
    draft=Path(sys.argv[3]) if len(sys.argv)>3 and sys.argv[3] else None
    try:check(model,draft)
    except (OSError,ValueError,KeyError,TypeError,struct.error) as e:
        print(f'GLM 모델 파일이 없거나 불완전합니다 ({host}): {e}',file=sys.stderr)
        print('복구: 설정 > 시스템 > 모델 준비에서 GLM을 선택하고 "모델만 준비"를 실행한 뒤 다시 시작하세요. 터미널: ./manage.sh model',file=sys.stderr)
        return 1
    print(f'{host}: GLM NVFP4 파일 확인 완료')
    return 0

if __name__=='__main__':sys.exit(main())
