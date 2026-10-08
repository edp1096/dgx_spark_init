<script>
  export let jobs
  export let openSettings
  export let refreshUserLoras
  export let tab
</script>

<nav>
  <button class:active={tab === 'image'} onclick={() => { tab = 'image'; refreshUserLoras() }}>이미지</button>
  <button class:active={tab === 'video'} onclick={() => tab = 'video'}>영상</button>
  <button class:active={tab === 'speech'} onclick={() => tab = 'speech'}>음성</button>
  <button class:active={tab === 'recognition'} onclick={() => tab = 'recognition'}>받아쓰기</button>
  <button class:active={tab === 'lora'} onclick={() => tab = 'lora'}>LoRA</button>
  <button class:active={tab === 'history'} onclick={() => tab = 'history'}>기록 <b>{jobs.length}</b></button>
  <button class:active={tab === 'settings'} onclick={openSettings}>설정</button>
</nav>
