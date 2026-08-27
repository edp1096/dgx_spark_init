<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  let releaseScroll = null

  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())
</script>

{#if open}
  <div class="runtime-info-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) open = false }}>
    <div class="runtime-info-modal" role="dialog" aria-modal="true" aria-label="모델 내부 조정 설명">
      <header><div><strong>모델 내부 조정</strong><small>필터 벡터는 한 번에 하나를 고르고 텍스트 조건은 필요할 때 함께 사용합니다.</small></div><button type="button" aria-label="닫기" onclick={() => open = false}>×</button></header>
      <div class="runtime-info-content">
        <article><strong>준수 강화 · skc3vo</strong><p>text-fusion projector 전체를 조절하는 rank-1 벡터입니다. 세부 지시를 더 직접적으로 따르게 하며 기본 강도는 0.05입니다.</p><a href="https://www.reddit.com/r/StableDiffusion/comments/1ueacq2/comment/otix1aa/" target="_blank" rel="noreferrer">출처 ↗</a></article>
        <article><strong>균형 · 2-vector</strong><p>Fedor 구현으로 projector의 두 필터 축만 완화합니다. 먼저 1.0에서 시작하고 부족할 때 2.0까지 올립니다.</p><a href="https://github.com/CliffNodes/fedor_bypass" target="_blank" rel="noreferrer">출처 ↗</a></article>
        <article><strong>강함 · 3-vector</strong><p>2-vector에 세 번째 필터 축을 더한 강한 대안입니다. 2-vector가 부족할 때 전환하며 두 방식을 중첩하지 않습니다.</p><a href="https://huggingface.co/uzumix/krea2filterbypass3.safetensors" target="_blank" rel="noreferrer">출처 ↗</a></article>
        <article><strong>프롬프트 준수 강화 · Krea2T</strong><p>Krea 2의 text-fusion 경로와 결합된 텍스트 토큰 비중을 조절해 객체 수, 배치, 관계 같은 지시를 더 강하게 전달합니다.</p><a href="https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer" target="_blank" rel="noreferrer">출처 ↗</a></article>
      </div>
    </div>
  </div>
{/if}
