<script>
  import { strToU8, zipSync } from 'fflate';

  export let artifacts = [];
  export let selectedId = '';
  export let onSelect = () => {};
  export let onClose = () => {};
  export let onStartResize = () => {};

  let mode = 'preview';
  let sourceFileName = 'index.html';
  let reloadKey = 0;
  $: selected = artifacts.find((item) => item.id === selectedId) || artifacts[0];
  $: sourceFiles = selected?.files || [];
  $: if (sourceFiles.length && !sourceFiles.some((item) => item.name === sourceFileName)) sourceFileName = sourceFiles[0].name;
  $: sourceFile = sourceFiles.find((item) => item.name === sourceFileName) || sourceFiles[0];

  function safeTitle() {
    return selected?.title.replace(/[^\p{L}\p{N}._-]+/gu, '-') || 'artifact';
  }

  function saveBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function downloadHTML() {
    if (!selected) return;
    saveBlob(new Blob([selected.document], { type: 'text/html;charset=utf-8' }), `${safeTitle()}.html`);
  }

  function downloadProject() {
    if (!selected?.files?.length) return;
    const files = Object.fromEntries(selected.files.map((item) => [item.name, strToU8(item.source)]));
    saveBlob(new Blob([zipSync(files, { level: 6 })], { type: 'application/zip' }), `${safeTitle()}.zip`);
  }

  function openWindow() {
    if (!selected) return;
    const blob = new Blob([selected.document], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank', 'noopener,noreferrer');
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }
</script>

<aside class="artifact-panel" aria-label="웹 생성물">
  <button class="artifact-resize" aria-label="생성물 패널 크기 조절" onpointerdown={onStartResize}></button>
  <header class="artifact-header">
    <div><strong>아티팩트</strong><small>격리된 웹 미리보기</small></div>
    <div class="artifact-header-actions">
      <button onclick={downloadHTML} title="실행형 단일 HTML 다운로드" aria-label="단일 HTML 다운로드">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v11m0 0 4-4m-4 4-4-4M5 15v4h14v-4" /></svg>
      </button>
      <button onclick={downloadProject} title="분리된 프로젝트 ZIP 다운로드" aria-label="프로젝트 ZIP 다운로드">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h6l2 2h8v10H4V7Zm8-3v9m0 0 3-3m-3 3-3-3" /></svg>
      </button>
      <button onclick={openWindow} title="새 창에서 열기" aria-label="새 창에서 열기">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 5h5v5m0-5-8 8M10 7H5v12h12v-5" /></svg>
      </button>
      <button onclick={onClose} aria-label="생성물 닫기">×</button>
    </div>
  </header>
  {#if artifacts.length > 1}
    <nav class="artifact-list" aria-label="생성물 목록">
      {#each artifacts as artifact}
        <button class:active={artifact.id === selected?.id} onclick={() => onSelect(artifact.id)}>{artifact.title}</button>
      {/each}
    </nav>
  {/if}
  <div class="artifact-toolbar">
    <div><button class:active={mode === 'preview'} onclick={() => mode = 'preview'}>미리보기</button><button class:active={mode === 'source'} onclick={() => mode = 'source'}>소스</button></div>
    {#if mode === 'preview'}<button onclick={() => reloadKey += 1} title="미리보기 다시 실행">↻ 새로고침</button>{/if}
  </div>
  {#if mode === 'source' && sourceFiles.length > 1}
    <nav class="artifact-source-files" aria-label="소스 파일">
      {#each sourceFiles as file}
        <button class:active={file.name === sourceFile?.name} onclick={() => sourceFileName = file.name}>{file.name}</button>
      {/each}
    </nav>
  {/if}
  <div class="artifact-stage">
    {#if selected}
      {#if mode === 'preview'}
        {#key `${selected.id}:${reloadKey}`}
          <iframe title={selected.title} srcdoc={selected.document} sandbox="allow-scripts allow-forms allow-modals"></iframe>
        {/key}
      {:else}
        <pre><code>{sourceFile?.source || selected.document}</code></pre>
      {/if}
    {:else}
      <div class="artifact-empty">표시할 웹 생성물이 없습니다.</div>
    {/if}
  </div>
  <footer class="artifact-security">스크립트는 SparkTalk과 분리되며 외부 통신이 차단됩니다.</footer>
</aside>
