<script>
  import { onMount, tick } from 'svelte'
  import { api } from './api.js'
  import SparkBolt from './SparkBolt.svelte'

  export let state = {}
  export let onActions = () => {}
  export let onExecute = async () => ''
  export let getVisualContext = async () => null

  const confirmationLabels = {
    image: '이미지 생성', video: '영상 생성', speech: '음성 생성', recognition: '자막 작업 시작'
  }

  let open = false
  let input = ''
  let sending = false
  let messages = []
  let messageList
  let inputElement

  function messageID() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
    return `message-${Date.now()}-${Math.random().toString(36).slice(2)}`
  }

  onMount(() => {
    const visualViewport = window.visualViewport
    const syncViewport = () => {
      const height = visualViewport?.height || window.innerHeight
      const top = visualViewport?.offsetTop || 0
      document.documentElement.style.setProperty('--spark-media-viewport-height', `${Math.round(height)}px`)
      document.documentElement.style.setProperty('--spark-media-viewport-top', `${Math.round(top)}px`)
    }
    syncViewport()
    window.addEventListener('resize', syncViewport)
    window.addEventListener('orientationchange', syncViewport)
    visualViewport?.addEventListener('resize', syncViewport)
    visualViewport?.addEventListener('scroll', syncViewport)
    try {
      const saved = JSON.parse(localStorage.getItem('spark-media-assistant') || '[]')
      if (Array.isArray(saved) && saved.length) messages = saved.slice(-20)
    } catch {}
    return () => {
      window.removeEventListener('resize', syncViewport)
      window.removeEventListener('orientationchange', syncViewport)
      visualViewport?.removeEventListener('resize', syncViewport)
      visualViewport?.removeEventListener('scroll', syncViewport)
      document.documentElement.style.removeProperty('--spark-media-viewport-height')
      document.documentElement.style.removeProperty('--spark-media-viewport-top')
    }
  })

  function saveMessages() {
    try {
      localStorage.setItem('spark-media-assistant', JSON.stringify(messages.slice(-20)))
    } catch {}
  }

  async function scrollToLatest() {
    await tick()
    if (messageList) messageList.scrollTop = messageList.scrollHeight
  }

  function clearChat() {
    messages = []
    saveMessages()
  }

  async function send() {
    const content = input.trim()
    if (!content || sending) return
    const userMessage = { id: messageID(), role: 'user', content }
    messages = [...messages, userMessage]
    input = ''
    await tick()
    resizeInput(inputElement)
    sending = true
    saveMessages()
    scrollToLatest()
    try {
      const requestMessages = messages
        .filter((message) => !message.status)
        .slice(-16)
        .map(({ role, content: text }) => ({ role, content: text }))
      const visualContext = await getVisualContext(content)
      const result = await api.assistantChat({ messages: requestMessages, state, visual_context: visualContext })
      onActions(result.actions || [])
      messages = [...messages, {
        id: messageID(), role: 'assistant', content: result.reply,
        confirmation: result.confirmation || '', visionUsed: Boolean(result.vision_used), executing: false, executed: false
      }]
    } catch (cause) {
      messages = [...messages, {
        id: messageID(), role: 'assistant', status: 'error',
        content: `요청을 처리하지 못했습니다. ${cause.message}`
      }]
    } finally {
      sending = false
      saveMessages()
      scrollToLatest()
    }
  }

  function handleInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      send()
    }
  }

  function resizeInput(element) {
    if (!element) return
    element.style.height = 'auto'
    const style = getComputedStyle(element)
    const lineHeight = Number.parseFloat(style.lineHeight) || 19
    const padding = (Number.parseFloat(style.paddingTop) || 0) + (Number.parseFloat(style.paddingBottom) || 0)
    const minimum = lineHeight + padding
    const maximum = lineHeight * 5 + padding
    element.style.height = `${Math.min(Math.max(element.scrollHeight, minimum), maximum)}px`
    element.style.overflowY = element.scrollHeight > maximum ? 'auto' : 'hidden'
  }

  async function execute(message) {
    if (!message.confirmation || message.executing || message.executed) return
    message.executing = true
    messages = [...messages]
    try {
      const status = await onExecute(message.confirmation)
      message.executed = true
      messages = [...messages, {
        id: messageID(), role: 'assistant', status: 'notice',
        content: status || '작업을 요청했습니다.'
      }]
    } catch (cause) {
      messages = [...messages, {
        id: messageID(), role: 'assistant', status: 'error',
        content: cause.message || '작업을 시작하지 못했습니다.'
      }]
    } finally {
      message.executing = false
      messages = [...messages]
      saveMessages()
      scrollToLatest()
    }
  }
</script>

<svelte:window onkeydown={(event) => { if (open && event.key === 'Escape') open = false }} />

{#if open}
  <div class="assistant-chat" role="dialog" aria-label="Aide 대화 도우미">
    <header>
      <div class="assistant-avatar"><SparkBolt label="Aide" /></div>
      <div class="assistant-heading"><strong>Aide</strong><small><i></i> Gemma 4 12B · 로컬</small></div>
      <button type="button" class="assistant-clear" title="대화 내역 비우기" aria-label="대화 내역 비우기" onclick={clearChat}>
        비우기
      </button>
      <button type="button" class="assistant-close" aria-label="대화창 닫기" onclick={() => open = false}>×</button>
    </header>
    <div class="assistant-messages" bind:this={messageList} aria-live="polite">
      {#each messages as message (message.id)}
        <article class:user={message.role === 'user'} class:error={message.status === 'error'} class:notice={message.status === 'notice'}>
          <p>{message.content}</p>
          {#if message.visionUsed}<small class="assistant-vision-note">시각 확인 · 선택 이미지</small>{/if}
          {#if message.confirmation}
            <button type="button" class="assistant-confirm" disabled={message.executing || message.executed} onclick={() => execute(message)}>
              {message.executing ? '요청 중…' : message.executed ? '요청 완료' : confirmationLabels[message.confirmation] || '작업 시작'}
            </button>
          {/if}
        </article>
      {/each}
      {#if sending}<article class="assistant-typing" aria-label="답변 작성 중"><span></span><span></span><span></span></article>{/if}
    </div>
    <form class="assistant-input" onsubmit={(event) => { event.preventDefault(); send() }}>
      <textarea bind:this={inputElement} bind:value={input} oninput={(event) => resizeInput(event.currentTarget)} onkeydown={handleInputKeydown} rows="1" maxlength="4000" placeholder="예: 한강이 보이는 서울 야경을 16:9로 만들어줘"></textarea>
      <button type="submit" aria-label="보내기" disabled={sending || !input.trim()}>➤</button>
    </form>
    <small class="assistant-footnote">설정은 즉시 반영되고 생성은 확인 후 시작됩니다.</small>
  </div>
{/if}

{#if !open}
  <button type="button" class="assistant-launcher" aria-expanded="false" aria-label="Aide 열기" onclick={() => { open = true; scrollToLatest() }}>
    <SparkBolt label="AI 대화 열기" />
  </button>
{/if}
