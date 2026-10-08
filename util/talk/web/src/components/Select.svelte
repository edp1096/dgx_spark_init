<script>
  import { afterUpdate, onMount, tick, getContext } from 'svelte';
  export let value = undefined;
  export let disabled = false;
  export let onchange = undefined;
  export let oninput = undefined;
  export let id = undefined;
  const field = getContext('settings-field');
  let source, trigger, menu;
  let items = [];
  let open = false;
  let active = -1;
  let label = '';
  let search = '', searchTime = 0;
  const uid = `select-${Math.random().toString(36).slice(2)}`;
  $: selected = items.find(item => Object.is(item.value, value)) || items.find(item => item.selected) || items[0];
  function sync() {
    if (!source) return;
    const next = Array.from(source.options, (option, index) => ({
      index, value: '__value' in option ? option.__value : option.value,
      text: option.textContent || '', disabled: option.disabled || option.parentElement?.disabled || false,
      selected: option.selected,
    }));
    if (next.length !== items.length || next.some((item, i) => Object.keys(item).some(key => item[key] !== items[i]?.[key]))) items = next;
    const owner = trigger?.closest('label');
    const inlineLabel = owner ? Array.from(owner.childNodes).filter(n => n.nodeType === 3 || (n.nodeName === 'SPAN' && !n.hidden)).map(n => n.textContent).join(' ').trim() : '';
    label = $$restProps['aria-label'] || field?.title || inlineLabel || trigger?.labels?.[0]?.textContent?.trim() || '';
  }
  onMount(() => {
    const owner = trigger.closest('label');
    const oldFor = owner?.getAttribute('for');
    if (owner && !oldFor) owner.htmlFor = trigger.id;
    sync();
    const observer = new MutationObserver(sync);
    observer.observe(source, { subtree: true, childList: true, characterData: true, attributes: true });
    const reposition = event => { if (open && !menu.contains(event.target)) position(); };
    window.addEventListener('resize', reposition);
    document.addEventListener('scroll', reposition, true);
    return () => { observer.disconnect(); if (owner && !oldFor && owner.htmlFor === trigger.id) owner.removeAttribute('for'); window.removeEventListener('resize', reposition); document.removeEventListener('scroll', reposition, true); };
  });
  afterUpdate(sync);
  function position() {
    if (disabled || !items.length) return;
    const rect = trigger.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom - 8;
    const spaceAbove = rect.top - 8;
    const height = Math.min(280, Math.max(spaceBelow, spaceAbove));
    menu.style.setProperty('--menu-width', `${Math.min(Math.max(rect.width, 180), window.innerWidth - 16)}px`);
    menu.style.setProperty('--menu-left', `${Math.max(8, Math.min(rect.left, window.innerWidth - Math.max(rect.width, 180) - 8))}px`);
    menu.style.setProperty('--menu-height', `${height}px`);
    menu.style.top = spaceBelow >= Math.min(280, spaceAbove) ? `${rect.bottom + 4}px` : 'auto';
    menu.style.bottom = spaceBelow >= Math.min(280, spaceAbove) ? 'auto' : `${window.innerHeight - rect.top + 4}px`;
    if (!open) active = selected && !selected.disabled ? selected.index : items.findIndex(item => !item.disabled);

  }
  function show() { if (!disabled && items.length) trigger.click(); }
  function close() { menu?.hidePopover(); open = false; }
  function focusOption() { menu?.querySelector(`[data-index="${active}"]`)?.focus({ preventScroll: true }); menu?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: 'nearest' }); }
  function choose(item) {
    if (!item || item.disabled) return;
    close();
    if (!Object.is(value, item.value)) {
      value = item.value;
      source.selectedIndex = item.index;
      source.dispatchEvent(new Event('input', { bubbles: true }));
      source.dispatchEvent(new Event('change', { bubbles: true }));
    }
    trigger.focus({ preventScroll: true });
  }
  function changed(event) { sync(); onchange?.(event); }
  function keydown(event) {
    if (disabled) return;
    const eligible = items.filter(item => !item.disabled);
    if (event.key === 'Escape') { close(); trigger.focus(); return; }
    if (event.key === 'Tab') { close(); return; }
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      if (!open) { show(); return; }
      const index = eligible.findIndex(item => item.index === active);
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? eligible.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + eligible.length) % eligible.length;
      active = eligible[next]?.index ?? -1; focusOption();
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (open) choose(items[active]); else show();
    } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey) {
      search = (Date.now() - searchTime < 700 ? search : '') + event.key.toLocaleLowerCase(); searchTime = Date.now();
      const match = eligible.find(item => item.text.toLocaleLowerCase().startsWith(search));
      if (match) { if (!open) show(); active = match.index; tick().then(focusOption); }
    }
  }
</script>

<span hidden aria-hidden="true"><select id={`${uid}-source`} bind:this={source} bind:value {disabled} onchange={changed} {oninput} tabindex="-1" aria-hidden="true" aria-label="internal-select-source" style="display:none"><slot /></select></span>
<button {...$$restProps} id={id || field?.id || `${uid}-trigger`} bind:this={trigger} type="button" class={`select-trigger ${$$restProps.class || ''}`} role="combobox" data-select-source={`${uid}-source`} data-value={String(value ?? '')} aria-label={label || undefined} aria-expanded={open} aria-controls={uid} aria-haspopup="listbox" {disabled} popovertarget={uid} onkeydown={keydown}>
  <span>{selected?.text || ''}</span><span aria-hidden="true">⌄</span>
</button>
<div bind:this={menu} id={uid} class="select-menu" popover="auto" role="listbox" aria-label="선택 항목" onbeforetoggle={event => { if (event.newState === 'open') position(); }} ontoggle={event => { open = event.newState === 'open'; if (open) tick().then(focusOption); }} onkeydown={keydown}>
  {#each items as item}
    <button type="button" role="option" data-index={item.index} aria-selected={Object.is(item.value, value)} disabled={item.disabled} tabindex="-1" onclick={() => choose(item)}>{item.text}</button>
  {/each}
</div>

<style>
  .select-trigger { width:100%; min-width:0; height:36px; display:flex; align-items:center; justify-content:space-between; gap:12px; padding:0 10px; border:1px solid #343948; border-radius:8px; color:#edf0f5; background:#0e1117; font:inherit; text-align:left; cursor:pointer; }
  .select-trigger > span:first-child { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .select-trigger:focus-visible { outline:2px solid #7399ff; outline-offset:2px; }
  .select-trigger:disabled { opacity:.5; cursor:default; }
  .select-menu { position:fixed; inset:auto; left:var(--menu-left); width:var(--menu-width); max-height:var(--menu-height); margin:0; padding:4px; overflow:auto; box-sizing:border-box; border:1px solid #48546b; border-radius:8px; background:#151b26; color:#edf0f5; box-shadow:0 8px 24px #0006; }
  .select-menu button { display:block; width:100%; padding:9px 10px; border:0; border-radius:5px; text-align:left; background:transparent; color:inherit; font:inherit; white-space:normal; cursor:pointer; }
  .select-menu button[aria-selected="true"] { background:#29374e; }
  .select-menu button:hover, .select-menu button:focus-visible { background:#344666; outline:0; }
  .select-menu button:disabled { opacity:.45; cursor:default; }
  :global(html[data-theme="light"]) .select-trigger { color:#2f3948; background:white; border-color:#cfd6e1; }
  :global(html[data-theme="light"]) .select-menu { color:#263347; background:white; border-color:#cfd6e1; }
  :global(html[data-theme="light"]) .select-menu button[aria-selected="true"] { background:#dce7f7; }
  :global(html[data-theme="light"]) .select-menu button:hover, :global(html[data-theme="light"]) .select-menu button:focus-visible { background:#cbdcf4; }
</style>
