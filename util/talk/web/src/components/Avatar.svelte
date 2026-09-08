<script>
  import { avatarURL } from '../lib/avatars.js';

  export let value = '';
  export let fallback = 'spark';
  export let alt = '';
</script>

<img class="avatar-picture" src={avatarURL(value, fallback)} {alt} />
