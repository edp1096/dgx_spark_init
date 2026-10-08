export function isGenerationTool(tool) {
  return ['video_generate', 'image_generate', 'flux2_generate'].includes(tool?.name);
}

const stages = {queued:'생성 작업 대기', preparing:'생성 서비스 준비', encode:'프롬프트 인코딩', release_encoder:'인코더 메모리 정리', sample:'샘플링', release_sample_workspace:'샘플링 메모리 정리', decode_video:'영상 디코딩', release_video_vae:'디코더 메모리 정리', decode_audio:'음성 디코딩', release_audio_vae:'음성 디코더 메모리 정리', save:'생성 파일 저장', attaching:'대화에 저장', attached:'대화 저장 완료'};
const duration = seconds => { const n=Math.max(0,Math.round(seconds || 0));return `${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`; };

export function generationProgressLabel(tool) {
  const p=tool?.progress;
  if(!p)return tool?.name==='video_generate'?'영상 생성 준비 중…':'이미지 생성 준비 중…';
  const stage=p.event==='idle'?'생성 파일 준비 완료':p.event==='failed'?'생성 실패':p.event==='conditioning_cache_hit'?'프롬프트 캐시 재사용':p.kind==='qwim' && p.stage==='decode_video'?'이미지 디코딩':stages[p.stage] || '생성 진행 중';
  const step=p.total>0 && p.step!==null && p.step!==undefined?` ${p.step}/${p.total}`:'';
  let eta='예상 시간 계산 중';
  if(p.eta_seconds!==null && p.eta_seconds!==undefined && p.eta_scope)eta=`${p.eta_scope==='stage'?'현재 단계':'전체'} 예상 ${duration(p.eta_seconds)} 남음`;
  if(p.event==='idle' || p.event==='failed' || p.stage==='attached')eta='';
  return [stage+step,eta,`경과 ${duration(p.elapsed_seconds)}`].filter(Boolean).join(' · ');
}
