<script>
  export let appearance;

  const choices = [
    { id: 'dark', label: '다크', icon: '☾' },
    { id: 'light', label: '라이트', icon: '☀' },
    { id: 'system', label: '시스템 설정 따름', icon: '◐' },
  ];

  function choose(theme) {
    appearance = { ...appearance, theme };
  }
</script>

<fieldset class="theme-settings">
  <legend>테마</legend>
  <div class="theme-choice" role="radiogroup" aria-label="테마">
    {#each choices as choice}
      <button type="button" role="radio" aria-checked={appearance.theme === choice.id} class:selected={appearance.theme === choice.id} onclick={() => choose(choice.id)}>
        <span aria-hidden="true">{choice.icon}</span><strong>{choice.label}</strong>
      </button>
    {/each}
  </div>
  <small>시스템 설정은 운영체제의 다크 모드 변경을 자동으로 따릅니다.</small>
</fieldset>
