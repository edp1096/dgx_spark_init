<script>
  export let index = 0
  export let total = 2
  export let prompt = ''
  export let enhancedPrompt = ''
  export let sceneTitle = ''
  export let onRemove = () => {}
  export let onPrompt = () => {}
  export let onMove = () => {}
</script>

<li class:planned={Boolean(enhancedPrompt)}>
  <div class="image-sequence-scene-heading">
    <span>{index + 1}</span>
    <label for={`sequence-scene-${index}`}>{sceneTitle || `장면 ${index + 1}`}</label>
    <div class="image-sequence-order-actions">
      <button type="button" aria-label={`장면 ${index + 1} 위로 이동`} title="위로" disabled={index === 0} onclick={() => onMove(-1)}>↑</button>
      <button type="button" aria-label={`장면 ${index + 1} 아래로 이동`} title="아래로" disabled={index === total - 1} onclick={() => onMove(1)}>↓</button>
      <button type="button" class="remove" aria-label={`장면 ${index + 1} 제거`} title="삭제" disabled={total <= 2} onclick={onRemove}>×</button>
    </div>
  </div>
  <textarea id={`sequence-scene-${index}`} rows="3" value={prompt} placeholder="이 장면만 보아도 완성된 그림이 되도록 인물·장소·행동을 적으세요." oninput={(event) => onPrompt(event.currentTarget.value)}></textarea>
  {#if enhancedPrompt}
    <details class="image-sequence-enhanced"><summary><span>독립 생성용 프롬프트 준비됨</span><b>향상 프롬프트</b></summary><p>{enhancedPrompt}</p></details>
  {/if}
</li>
