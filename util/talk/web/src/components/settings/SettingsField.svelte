<script>
  import { onMount } from 'svelte';
  import SettingsHelp from './SettingsHelp.svelte';
  import { createClientID } from '../../lib/client-id.js';
  export let title;
  let controlHost;
  const controlID = `settings-field-${createClientID()}`;
  onMount(() => {
    const control = controlHost.querySelector('input, select');
    if (control) control.id = controlID;
  });
</script>

<div class="settings-help-field">
  <div class="settings-field-caption"><label for={controlID}>{title}</label><SettingsHelp {title}><slot name="help" /></SettingsHelp></div>
  <div class="settings-field-control" bind:this={controlHost}><slot /></div>
</div>
