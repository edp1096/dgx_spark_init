<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'
  import { normalizeTagName, tagKey } from './lib/jobLists.js'

  export let job
  export let availableTags = []
  export let saving = false
  export let onSave = async () => true

  let open = false
  let draft = []
  let input = ''
  let message = ''
  let releaseScroll = null

  $: draftKeys = new Set(draft.map(tagKey))
  $: suggestions = availableTags.filter((tag) => !draftKeys.has(tagKey(tag.name)) && (!input.trim() || tag.name.toLocaleLowerCase().includes(input.trim().toLocaleLowerCase()))).slice(0, 12)
  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())

  function openEditor() {
    draft = [...(job.tags || [])]
    input = ''
    message = ''
    open = true
  }

  function close() {
    if (saving) return
    open = false
  }

  function addTag(value = input) {
    const name = normalizeTagName(value)
    if (!name) return
    if (name.includes(',')) {
      message = '쉼표는 태그 이름에 사용할 수 없습니다.'
      return
    }
    if ([...name].length > 32) {
      message = '태그 이름은 32자까지 입력할 수 있습니다.'
      return
    }
    if (draft.length >= 24) {
      message = '한 항목에는 태그를 24개까지 붙일 수 있습니다.'
      return
    }
    if (!draftKeys.has(tagKey(name))) draft = [...draft, name]
    input = ''
    message = ''
  }

  function removeTag(name) {
    draft = draft.filter((tag) => tagKey(tag) !== tagKey(name))
  }

  async function save() {
    if (input.trim()) addTag()
    const succeeded = await onSave(job, draft)
    if (succeeded !== false) open = false
  }
</script>

<svelte:window onkeydown={(event) => { if (open && event.key === 'Escape') close() }} />

<div class="job-tags" aria-label="결과 태그">
  {#each job.tags || [] as tag}<span title={tag}>#{tag}</span>{/each}
  <button type="button" disabled={saving} onclick={openEditor}>{saving ? '저장 중…' : job.tags?.length ? '편집' : '+ 태그'}</button>
</div>

{#if open}
  <div class="tag-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) close() }}>
    <section class="tag-editor-modal" aria-label="결과 태그 편집">
      <header><div><strong>태그 편집</strong><small>#{job.id} · Enter로 추가</small></div><button type="button" aria-label="닫기" disabled={saving} onclick={close}>×</button></header>
      <div class="tag-editor-body">
        <div class="tag-editor-input"><input bind:value={input} maxlength="32" placeholder="새 태그 입력" onkeydown={(event) => { if (event.key === 'Enter' || event.key === ',') { event.preventDefault(); addTag() } }}><button type="button" onclick={() => addTag()}>추가</button></div>
        {#if message}<em>{message}</em>{/if}
        <div class="tag-editor-selected">
          {#each draft as tag}<button type="button" title={`${tag} 제거`} onclick={() => removeTag(tag)}>#{tag}<span>×</span></button>{/each}
          {#if !draft.length}<p>아직 선택된 태그가 없습니다.</p>{/if}
        </div>
        {#if suggestions.length}
          <div class="tag-editor-suggestions"><small>기존 태그</small><div>{#each suggestions as tag}<button type="button" onclick={() => addTag(tag.name)}>#{tag.name}<span>{tag.count}</span></button>{/each}</div></div>
        {/if}
      </div>
      <footer><button type="button" disabled={saving} onclick={close}>취소</button><button type="button" class="primary" disabled={saving} onclick={save}>{saving ? '저장 중…' : '저장'}</button></footer>
    </section>
  </div>
{/if}
