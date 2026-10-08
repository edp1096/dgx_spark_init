<script>
  export let label = ''
</script>

<svg class="spark-bolt" viewBox="0 0 64 64" role={label ? 'img' : undefined} aria-label={label || undefined} aria-hidden={label ? undefined : 'true'}>
  <defs>
    <linearGradient id="spark-bolt-fill" x1="18" y1="7" x2="45" y2="57" gradientUnits="userSpaceOnUse">
      <stop stop-color="#fffde8" />
      <stop offset=".36" stop-color="#ffd640" />
      <stop offset="1" stop-color="#ff8a00" />
    </linearGradient>
    <filter id="spark-bolt-glow" x="-80%" y="-60%" width="260%" height="220%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
    </filter>
  </defs>
  <path class="spark-bolt-halo" d="M38 3 13 36h16l-5 25 27-36H35z" />
  <path class="spark-bolt-core" d="M38 3 13 36h16l-5 25 27-36H35z" />
  <path class="spark-bolt-cut" d="m35 14-14 18h13l-4 17 13-20H31z" />
</svg>

<style>
  .spark-bolt { display: block; width: 100%; height: 100%; overflow: visible; }
  .spark-bolt-halo { fill: #ffc400; opacity: .72; filter: url(#spark-bolt-glow); }
  .spark-bolt-core { fill: url(#spark-bolt-fill); stroke: #fff7a8; stroke-width: 1.15; stroke-linejoin: round; }
  .spark-bolt-cut { fill: #fff; opacity: .72; }
</style>
