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


def check(model,draft):
    json.loads((model/'config.json').read_text())
    index=json.loads((model/'model.safetensors.index.json').read_text())
    names=set(index['weight_map'].values())
    if len(names)<120:raise ValueError(f'GLM EXL3 shard 목록 불완전: {model}')
    for name in sorted(names):
        if Path(name).name!=name:raise ValueError('잘못된 shard 경로')
        check_tensor_file(model/name)
    json.loads((draft/'config.json').read_text())
    check_tensor_file(draft/'model.safetensors')


def main():
    model,draft=map(Path,sys.argv[1:3]);host=sys.argv[3] if len(sys.argv)>3 else '호스트'
    try:check(model,draft)
    except (OSError,ValueError,KeyError,TypeError,struct.error) as e:
        print(f'GLM 모델 파일이 없거나 불완전합니다 ({host}): {e}',file=sys.stderr)
        print('복구: 설정 > 시스템 > 모델 준비에서 GLM을 선택하고 "모델만 준비"를 실행한 뒤 다시 시작하세요. 터미널: ./manage.sh model',file=sys.stderr)
        return 1
    print(f'{host}: GLM EXL3·DFlash2 파일 확인 완료')
    return 0

if __name__=='__main__':sys.exit(main())
