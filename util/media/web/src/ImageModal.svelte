<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'
  import { api } from './api.js'

  export let image = null
  export let onClose = () => {}
  export let onGarmentExtract = null
  export let onFaceSwap = null

  let releaseScroll = null
  let metadataJobID = ''
  let exifOpen = false
  let exifLoading = false
  let exifError = ''
  let exifResult = null

  $: if ((image?.jobID || '') !== metadataJobID) {
    metadataJobID = image?.jobID || ''
    exifOpen = false
    exifLoading = false
    exifError = ''
    exifResult = null
  }

  async function toggleEXIF() {
    if (exifOpen) {
      exifOpen = false
      return
    }
    exifOpen = true
    if (!image?.jobID || exifResult || exifLoading) return
    exifLoading = true
    exifError = ''
    try {
      exifResult = await api.imageEXIF(image.jobID)
    } catch (error) {
      exifError = error.message
    } finally {
      exifLoading = false
    }
  }

  function modeLabel(mode) {
    return ({ create: 'Krea 2 생성', edit: '원본 수정', control: '구조 제어', detail_enhance: '디테일 강화', upscale: '고화질 확대', garment_extract: '의상 추출', face_swap: 'ReActor 얼굴 교체' })[mode] || mode || '—'
  }

  function formatCreatedAt(value) {
    if (!value) return '—'
    const date = new Date(value)
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
  }

  function unlockScroll() {
    releaseScroll?.()
    releaseScroll = null
  }

  $: {
    if (image && !releaseScroll) {
      releaseScroll = lockModalScroll()
    } else if (!image) {
      unlockScroll()
    }
  }

  onDestroy(unlockScroll)
</script>

<svelte:window onkeydown={(event) => { if (image && event.key === 'Escape') onClose() }} />

{#if image}
  <div class="image-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div class="image-modal" role="dialog" aria-modal="true" aria-label="이미지 크게 보기">
      <header><div><strong>{image.title || '생성 이미지'}</strong>{#if image.detail}<small title={image.detail}>{image.detail}</small>{/if}</div><button type="button" aria-label="닫기" onclick={onClose}>×</button></header>
      <div class="image-modal-stage">
        {#if exifOpen}
          <div class="image-exif-view">
            {#if exifLoading}<p class="image-exif-state">EXIF 정보를 읽는 중…</p>
            {:else if exifError}<p class="image-exif-state error-state">{exifError}</p>
            {:else if exifResult && !exifResult.embedded}<p class="image-exif-state">이 파일에는 SparkMediaPanel EXIF 정보가 없습니다.</p>
            {:else if exifResult?.metadata}
              {@const metadata = exifResult.metadata}
              <dl>
                {#if metadata.creator}<div><dt>제작자</dt><dd>{metadata.creator}</dd></div>{/if}
                {#if metadata.copyright}<div><dt>저작권</dt><dd>{metadata.copyright}</dd></div>{/if}
                {#if metadata.website}<div><dt>웹사이트·연락처</dt><dd>{metadata.website}</dd></div>{/if}
                <div><dt>모델</dt><dd>{metadata.model || '—'}</dd></div>
                <div><dt>작업</dt><dd>{modeLabel(metadata.mode)}</dd></div>
                <div><dt>크기</dt><dd>{metadata.width || '—'} × {metadata.height || '—'}</dd></div>
                <div><dt>시드</dt><dd>{metadata.seed ?? '—'}</dd></div>
                <div><dt>샘플러</dt><dd>{metadata.parameters?.sampler || '—'}</dd></div>
                <div><dt>스케줄러</dt><dd>{metadata.parameters?.scheduler || '—'}</dd></div>
                <div><dt>스텝</dt><dd>{metadata.parameters?.steps ?? '—'}</dd></div>
                <div><dt>생성 시각</dt><dd>{formatCreatedAt(metadata.created_at)}</dd></div>
                <div><dt>작업 ID</dt><dd>{metadata.job_id || '—'}</dd></div>
              </dl>
              {#if metadata.note}<section><strong>제작자 메모</strong><p>{metadata.note}</p></section>{/if}
              <section><strong>원본 프롬프트</strong><p>{metadata.prompt || '—'}</p></section>
              {#if metadata.effective_prompt}<section><strong>실제 생성 프롬프트</strong><p>{metadata.effective_prompt}</p></section>{/if}
              <details><summary>전체 생성 설정</summary><pre>{JSON.stringify(metadata.parameters || {}, null, 2)}</pre></details>
            {/if}
          </div>
        {:else}
          <img src={image.src} alt={image.title || '확대 이미지'}>
        {/if}
      </div>
      <footer><div><a href={image.src} target="_blank" rel="noreferrer">원본 파일 열기</a>{#if image.jobID}<button type="button" class:active={exifOpen} onclick={toggleEXIF}>{exifOpen ? '이미지 보기' : 'EXIF 보기'}</button>{/if}{#if image.jobID && onFaceSwap}<button type="button" onclick={() => onFaceSwap(image.jobID)}>얼굴 교체</button>{/if}{#if image.jobID && onGarmentExtract}<button type="button" onclick={() => onGarmentExtract(image.jobID)}>의상 추출</button>{/if}</div><button type="button" onclick={onClose}>닫기</button></footer>
    </div>
  </div>
{/if}
