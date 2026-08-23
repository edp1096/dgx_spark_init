<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let prompt = null
  export let onClose = () => {}
  let copied = false
  let copyFailed = false
  let releaseScroll = null

  function unlockScroll() {
    releaseScroll?.()
    releaseScroll = null
  }

  $: {
    if (prompt && !releaseScroll) {
      releaseScroll = lockModalScroll()
    } else if (!prompt) {
      unlockScroll()
    }
  }

  onDestroy(unlockScroll)

  $: if (!prompt) {
    copied = false
    copyFailed = false
  }

  function legacyCopy(text) {
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.setAttribute('readonly', '')
    textarea.style.position = 'fixed'
    textarea.style.left = '-9999px'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.focus()
    textarea.select()
    textarea.setSelectionRange(0, textarea.value.length)
    const succeeded = document.execCommand('copy')
    textarea.remove()
    return succeeded
  }

  function copyPrompt() {
    const text = prompt?.text || ''
    copied = false
    copyFailed = false
    copied = legacyCopy(text)
    copyFailed = !copied
  }
</script>

<svelte:window onkeydown={(event) => { if (prompt && event.key === 'Escape') onClose() }} />

{#if prompt}
  <div class="prompt-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section class="prompt-modal" aria-label="프롬프트 전체 보기">
      <header><div><strong>{prompt.title || '전체 프롬프트'}</strong><small>{prompt.detail || '생성에 사용한 원문'}</small></div><button type="button" aria-label="닫기" onclick={onClose}>×</button></header>
      <pre>{prompt.text}</pre>
      <footer><button type="button" class="copy-prompt" onclick={copyPrompt}>{copied ? '복사됨' : copyFailed ? '복사 실패' : '프롬프트 복사'}</button><button type="button" onclick={onClose}>닫기</button></footer>
    </section>
  </div>
{/if}
