<script>
  import { onMount } from 'svelte';
  import {
    collectKnowledgeSource, createKnowledgeCollection, deleteKnowledgeCollection, deleteKnowledgeDocument,
    knowledgeSourceURL, listKnowledgeCollections, listKnowledgeDocuments, ocrKnowledgeDocument,
    searchKnowledge, updateKnowledgeCollection, uploadKnowledgeDocument,
  } from '../../api.js';

  export let onnotify = () => {};
  export let health = null;
  export let onjobcreated = () => {};

  let collections = [];
  let selectedID = 0;
  let documents = [];
  let loading = true;
  let busy = false;
  let uploadInput;
  let searchQuery = '';
  let searchResults = [];
  let searched = false;
  let newCollection = { name: '', description: '' };
  let sourceURL = '';
  let sourceMode = 'auto';
  let discoveredLinks = [];
	let discoveredPublication = null;
  let documentsExpanded = true;
  let documentFilter = 'all';
  let progressRefreshing = false;

  $: selected = collections.find((item) => item.id === Number(selectedID));
  $: problemDocuments = documents.filter(isProblemDocument);
  $: visibleDocuments = documentFilter === 'problems' ? problemDocuments : documents;

  onMount(() => {
    loadCollections();
    const timer = setInterval(refreshOCRProgress, 1500);
    return () => clearInterval(timer);
  });

  async function loadCollections(preferredID = selectedID) {
    loading = true;
    try {
      collections = await listKnowledgeCollections();
      selectedID = collections.some((item) => item.id === Number(preferredID)) ? Number(preferredID) : (collections[0]?.id || 0);
      await loadDocuments();
    } catch (error) { onnotify(error.message, 'error'); }
    finally { loading = false; }
  }

  async function loadDocuments() {
    documents = selectedID ? await listKnowledgeDocuments(selectedID) : [];
    searchResults = [];
    searched = false;
  }

  async function selectCollection() {
    loading = true;
    discoveredLinks = [];
		discoveredPublication = null;
    try { await loadDocuments(); }
    catch (error) { onnotify(error.message, 'error'); }
    finally { loading = false; }
  }

  async function addCollection() {
    if (busy || !newCollection.name.trim()) return;
    busy = true;
    try {
      const created = await createKnowledgeCollection(newCollection);
      newCollection = { name: '', description: '' };
      await loadCollections(created.id);
      onnotify('지식 보관함을 추가했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { busy = false; }
  }

  async function saveCollection() {
    if (busy || !selected?.name?.trim()) return;
    busy = true;
    try {
      const updated = await updateKnowledgeCollection(selected.id, selected);
      collections = collections.map((item) => item.id === updated.id ? updated : item);
      onnotify('지식 보관함을 저장했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { busy = false; }
  }

  async function removeCollection() {
    if (busy || !selected || !confirm(`“${selected.name}” 보관함과 문서를 모두 삭제할까요?`)) return;
    busy = true;
    try {
      await deleteKnowledgeCollection(selected.id);
      await loadCollections();
      onnotify('지식 보관함을 삭제했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { busy = false; }
  }

  async function uploadFiles(event) {
    const files = [...(event.currentTarget.files || [])];
    event.currentTarget.value = '';
    if (!files.length || !selectedID || busy) return;
    busy = true;
    let completed = 0;
    try {
      for (const file of files) {
        await uploadKnowledgeDocument(selectedID, file);
        completed += 1;
      }
      await loadCollections(selectedID);
      onnotify(`지식 문서 ${completed}개를 가져왔습니다.`);
    } catch (error) {
      await loadCollections(selectedID);
      onnotify(`${completed}개 처리 후 중단: ${error.message}`, 'error');
    } finally { busy = false; }
  }

  async function collectURL() {
    if (busy || !selectedID || !sourceURL.trim()) return;
    busy = true;
    try {
      const result = await collectKnowledgeSource(selectedID, sourceURL.trim(), sourceMode);
      const created = result.document;
      discoveredLinks = Array.isArray(result.links) ? result.links : [];
		discoveredPublication = result.publication || null;
      sourceURL = '';
      await loadCollections(selectedID);
		if (discoveredPublication) {
			onnotify(`“${created.title}” 전자책 ${discoveredPublication.page_count}쪽을 감지했습니다. 가져오기 시작을 기다립니다.`);
			if (result.job) onjobcreated(result.job);
		}
		else onnotify(`“${created.title}” 자료를 가져왔습니다.`);
    } catch (error) {
      await loadCollections(selectedID);
      onnotify(error.message, 'error');
    } finally { busy = false; }
  }

  async function removeDocument(document) {
    if (busy || !confirm(`“${document.title}” 문서를 삭제할까요?`)) return;
    busy = true;
    try {
      await deleteKnowledgeDocument(document.id);
      await loadCollections(selectedID);
      onnotify('지식 문서를 삭제했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { busy = false; }
  }

  async function runOCR(document) {
    if (busy) return;
    busy = true;
    try {
      const started = await ocrKnowledgeDocument(document.id);
      documents = documents.map((item) => item.id === started.id ? started : item);
      if (started.status === 'ready') onnotify('문서 OCR과 색인을 완료했습니다.');
      else onnotify('OCR 작업을 시작했습니다. 화면을 이동해도 계속 처리됩니다.');
    } catch (error) {
      await loadCollections(selectedID);
      onnotify(error.message, 'error');
    } finally { busy = false; }
  }

  async function refreshOCRProgress() {
    if (progressRefreshing || !selectedID || !documents.some((item) => item.status === 'processing' && item.ocr_total_pages > 0)) return;
    progressRefreshing = true;
    const collectionID = Number(selectedID);
    const previous = new Map(documents.map((item) => [item.id, item.status]));
    try {
      const latest = await listKnowledgeDocuments(collectionID);
      if (Number(selectedID) !== collectionID) return;
      documents = latest;
      for (const item of latest) {
        if (previous.get(item.id) !== 'processing' || item.status === 'processing') continue;
        if (item.status === 'ready') onnotify(`“${item.title}” OCR과 색인을 완료했습니다.`);
        else if (item.error) onnotify(item.error, 'error');
      }
    } catch (error) {
      // Temporary polling failures are retried without replacing the current list.
    } finally { progressRefreshing = false; }
  }

  async function runSearch() {
    if (!searchQuery.trim() || busy) return;
    busy = true;
    try {
      searchResults = await searchKnowledge(searchQuery, selectedID, 20);
      searched = true;
    } catch (error) { onnotify(error.message, 'error'); }
    finally { busy = false; }
  }

  function statusLabel(status) {
    return ({ ready: '검색 가능', processing: '처리 중', paused: '가져오기 대기', canceled: '취소됨', needs_ocr: 'OCR 필요', failed: '처리 실패' })[status] || status;
  }

  function isProblemDocument(document) {
    return Boolean(document?.error) || ['failed', 'needs_ocr', 'canceled'].includes(document?.status);
  }

  function formatBytes(value) {
    if (!value) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    return `${(value / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
  }
</script>

<fieldset>
  <legend>지식 보관함</legend>
  {#if loading && !collections.length}
    <small>지식 보관함을 불러오는 중…</small>
  {:else}
    <div class="knowledge-picker">
      <label>현재 보관함<select bind:value={selectedID} onchange={selectCollection}>{#each collections as collection}<option value={collection.id}>{collection.name} · {collection.documents}</option>{/each}</select></label>
      <label class="knowledge-enabled"><input type="checkbox" bind:checked={selected.enabled} disabled={!selected} /> 대화 검색에 사용</label>
    </div>
    {#if selected}
      <div class="settings-form-row two">
        <label>이름<input bind:value={selected.name} maxlength="80" /></label>
        <label>설명<input bind:value={selected.description} maxlength="500" placeholder="선택 사항" /></label>
      </div>
      <div class="knowledge-actions"><button onclick={removeCollection} disabled={busy}>보관함 삭제</button><button class="primary" onclick={saveCollection} disabled={busy || !selected.name.trim()}>보관함 저장</button></div>
    {/if}
  {/if}
  <details class="knowledge-new">
    <summary>새 보관함 만들기</summary>
    <div class="settings-form-row two">
      <label>이름<input bind:value={newCollection.name} maxlength="80" placeholder="예: 학교 교과서" /></label>
      <label>설명<input bind:value={newCollection.description} maxlength="500" placeholder="선택 사항" /></label>
    </div>
    <button onclick={addCollection} disabled={busy || !newCollection.name.trim()}>추가</button>
  </details>
</fieldset>

<fieldset>
  <legend>문서</legend>
  <div class="knowledge-source">
    <input type="url" bind:value={sourceURL} onkeydown={(event) => event.key === 'Enter' && collectURL()} placeholder="웹 문서 주소" aria-label="웹 문서 주소" />
    <select bind:value={sourceMode} aria-label="수집 방식"><option value="auto">자동</option><option value="direct">직접 요청</option><option value="browser">브라우저</option></select>
    <button onclick={collectURL} disabled={busy || !selectedID || !sourceURL.trim()}>주소 가져오기</button>
  </div>
  <small>자동은 가벼운 직접 요청을 먼저 사용하고, 본문이 부족한 HTML만 격리된 Chromium으로 다시 읽습니다.{health ? ` Collector · ${health.status === 'ok' ? 'online' : 'offline'}` : ''}</small>
	{#if discoveredPublication}
		<div class="knowledge-publication"><strong>전자책 감지 · {discoveredPublication.title || '제목 없음'}</strong><small>{discoveredPublication.adapter} 어댑터 · {discoveredPublication.page_count}쪽</small><span>가져오기 작업에서 시작·중지·재개할 수 있습니다.</span></div>
	{/if}
  {#if discoveredLinks.length}
    <details class="knowledge-discovered">
      <summary>수집한 페이지에서 찾은 링크 {discoveredLinks.length}개</summary>
      <div>{#each discoveredLinks as link (link.url)}<article><a href={link.url} target="_blank" rel="noreferrer">{link.text || link.url}</a><button onclick={() => { sourceURL = link.url; sourceMode = 'auto'; }}>주소 선택</button></article>{/each}</div>
    </details>
  {/if}
  <div class="knowledge-file-row">
	<input class="knowledge-file-input" bind:this={uploadInput} type="file" multiple accept=".pdf,.txt,.md,.markdown,.html,.htm,.csv,.tsv,.json,.xml,.yaml,.yml,.toml,.js,.jsx,.ts,.tsx,.css,.scss,.py,.go,.rs,.java,.c,.h,.cpp,.hpp,.cs,.sh,.ps1,.sql,.ini,.conf,.log,.docx,.pptx,.xlsx,.odt,.odp,.ods,.epub,.hwpx,.png,.jpg,.jpeg,.webp" onchange={uploadFiles} />
  <button class="knowledge-upload" onclick={() => uploadInput?.click()} disabled={busy || !selectedID}>{busy ? '처리 중…' : '파일 가져오기'}</button>
	<small>PDF·텍스트·웹 문서·표·최신 오피스·EPUB·HWPX·이미지를 가져올 수 있습니다.</small>
  </div>
  <small>원본은 DB 옆의 전용 폴더에 보존합니다. 스캔 PDF·이미지는 OCR 대기 상태로 남습니다.</small>
  <div class="knowledge-document-toolbar">
    <button class="knowledge-document-toggle" type="button" aria-expanded={documentsExpanded} onclick={() => documentsExpanded = !documentsExpanded}>
      <span>{documentsExpanded ? '▾' : '▸'} 파일 목록</span><small>{documents.length}</small>
    </button>
    <div class="knowledge-document-filter" role="group" aria-label="파일 목록 필터">
      <button type="button" class:active={documentFilter === 'all'} aria-pressed={documentFilter === 'all'} onclick={() => documentFilter = 'all'}>전체</button>
      <button type="button" class:active={documentFilter === 'problems'} aria-pressed={documentFilter === 'problems'} onclick={() => documentFilter = 'problems'}>오류만{problemDocuments.length ? ` ${problemDocuments.length}` : ''}</button>
    </div>
  </div>
  {#if documentsExpanded}
    {#if !loading && !documents.length}
      <div class="knowledge-empty">이 보관함에 문서가 없습니다.</div>
    {:else if !loading && documentFilter === 'problems' && !visibleDocuments.length}
      <div class="knowledge-empty">문제가 표시된 문서가 없습니다.</div>
    {:else}
      <div class="knowledge-documents">
        {#each visibleDocuments as document (document.id)}
          <article class="knowledge-document">
            <div class="knowledge-document-main"><strong>{document.title}</strong><small>{document.source_name} · {formatBytes(document.size_bytes)}{document.page_count ? ` · ${document.page_count}쪽` : ''}{document.chunk_count ? ` · ${document.chunk_count}구간` : ''}</small>{#if document.ocr_total_pages > 0 && (document.status === 'processing' || document.ocr_processed_pages > 0)}<div class="knowledge-ocr-progress"><progress value={document.ocr_processed_pages} max={document.ocr_total_pages}></progress><small>{document.ocr_processed_pages}/{document.ocr_total_pages}쪽 완료</small></div>{/if}{#if document.error}<small class="knowledge-error" title={document.error}>{document.error}</small>{/if}</div>
            <span class:ready={document.status === 'ready'} class:warning={document.status === 'needs_ocr'} class:error={document.status === 'failed'}>{statusLabel(document.status)}</span>
            <div class="knowledge-document-actions">{#if document.source_url}<a href={document.source_url} target="_blank" rel="noreferrer">출처</a>{/if}<a href={knowledgeSourceURL(document.id)} target="_blank" rel="noreferrer">원문</a>{#if document.status === 'needs_ocr'}<button onclick={() => runOCR(document)} disabled={busy}>OCR 실행</button>{/if}<button onclick={() => removeDocument(document)} disabled={busy || document.status === 'processing'}>삭제</button></div>
          </article>
        {/each}
      </div>
    {/if}
  {/if}
</fieldset>

<fieldset>
  <legend>색인 확인</legend>
  <div class="knowledge-search"><input bind:value={searchQuery} onkeydown={(event) => event.key === 'Enter' && runSearch()} placeholder="현재 보관함 검색" /><button onclick={runSearch} disabled={busy || !searchQuery.trim()}>검색</button></div>
  {#if searched && !searchResults.length}<small>검색 결과가 없습니다.</small>{/if}
  {#if searchResults.length}
    <div class="knowledge-results">
      {#each searchResults as result (result.chunk_id)}
        <article><strong>{result.title}{result.page_start ? ` · ${result.page_start}쪽` : ''}</strong><p>{result.content}</p></article>
      {/each}
    </div>
  {/if}
</fieldset>
