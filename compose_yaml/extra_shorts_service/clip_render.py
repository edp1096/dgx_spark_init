"""Isolated process for a durable image-to-video production."""
import os
import sys
import time

import httpx

import clip_pipeline
import clip_store


def bridge(operation,payload):
    endpoint=os.environ.get('SHORTS_TALK_ENDPOINT','http://127.0.0.1:8585').rstrip('/')
    last_error=None
    for attempt in range(5):
        try:
            with httpx.Client(timeout=1800) as client:
                response=client.post(endpoint+'/api/clip-production/'+operation,json=payload)
                if response.status_code==200:
                    return response.json()
                if response.status_code not in (502,503,504):
                    raise RuntimeError(response.text[:2500])
                last_error=response.text[:2500]
        except httpx.TransportError as error:
            last_error=str(error)
        if attempt<4:
            job=clip_store.get(payload['job_id'])
            if job['canceled']:raise RuntimeError('제작이 중지됐습니다')
            clip_store.update(job['id'],'running',f'제작 서비스 연결 복구 중 {attempt+1}/4',job['progress'])
            time.sleep(2**(attempt+1))
    raise RuntimeError('제작 서비스 연결을 복구하지 못했습니다: '+str(last_error))


if __name__=='__main__':
    try:
        clip_pipeline.produce(sys.argv[1],bridge)
    except Exception as error:
        (clip_store.folder(sys.argv[1])/'error.txt').write_text(str(error))
        raise
