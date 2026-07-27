// Landing-page header treatment + subtle ASCII smoke over the hero logo.
// Works with Material's instant navigation via the document$ observable.
(function () {
  var COLS = 15;
  var ROWS = 12;
  // Light density ramp — deliberately stops short of heavy glyphs so the
  // plume stays a soft wisp rather than a solid block.
  var CHARS = " .·:;+*";
  var MAX_PARTICLES = 48;
  var TICK_MS = 70;
  // Soft splat so neighbouring cells connect into a continuous column
  // instead of reading as scattered dots.
  var SPLAT = [
    [0, 0, 1],
    [-1, 0, 0.4],
    [1, 0, 0.4],
    [0, -1, 0.3],
    [0, 1, 0.3],
    [-1, -1, 0.14],
    [-1, 1, 0.14],
  ];

  var smokeTimer = null;
  var particles = [];
  var smokeEl = null;

  function updateHeader() {
    var isLanding = !!document.querySelector(".hero");
    document.body.classList.toggle("landing-home", isLanding);
    document.body.classList.toggle(
      "landing-scrolled",
      isLanding && window.scrollY > 40
    );
  }

  function restartHeroEnter() {
    var nodes = document.querySelectorAll(
      ".hero-logo-wrap, .hero .hero-wordmark"
    );
    if (!nodes.length) return;
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].style.animation = "none";
    }
    void document.body.offsetWidth;
    for (var j = 0; j < nodes.length; j++) {
      nodes[j].style.animation = "";
    }
  }

  function prefersReducedMotion() {
    return (
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  }

  function stopAsciiSmoke() {
    if (smokeTimer) {
      clearInterval(smokeTimer);
      smokeTimer = null;
    }
    particles = [];
    smokeEl = null;
  }

  function spawnParticle(atBase) {
    particles.push({
      x: COLS * 0.5 - 0.7 + Math.random() * 1.4,
      // Stagger seeded particles up the column so the plume is continuous
      // from the first frame.
      y: atBase ? ROWS - 0.5 - Math.random() : Math.random() * ROWS,
      vx: (Math.random() - 0.5) * 0.14,
      vy: -0.16 - Math.random() * 0.16,
      life: 1,
      // Slow enough to survive the full climb up the taller column.
      decay: 0.007 + Math.random() * 0.008,
      dens: 0.45 + Math.random() * 0.35,
      wobble: Math.random() * Math.PI * 2,
    });
  }

  function renderAsciiSmoke(el) {
    var grid = [];
    var r, c, i, s, p, row, col, dens, idx;

    for (r = 0; r < ROWS; r++) {
      grid[r] = [];
      for (c = 0; c < COLS; c++) grid[r][c] = 0;
    }

    for (i = 0; i < particles.length; i++) {
      p = particles[i];
      row = Math.floor(p.y + 0.5);
      col = Math.floor(p.x + 0.5);
      dens = p.life * p.dens;
      for (s = 0; s < SPLAT.length; s++) {
        var rr = row + SPLAT[s][0];
        var cc = col + SPLAT[s][1];
        if (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS) continue;
        grid[rr][cc] += dens * SPLAT[s][2] * 0.6;
      }
    }

    var lines = [];
    for (r = 0; r < ROWS; r++) {
      var line = "";
      // Thin out toward the top so the plume dissipates as it rises.
      var rowFade = 0.5 + 0.5 * (r / (ROWS - 1));
      for (c = 0; c < COLS; c++) {
        // Cap below 1 so the densest glyph stays rare.
        dens = Math.min(0.92, grid[r][c] * rowFade);
        idx = Math.floor(dens * CHARS.length);
        line += CHARS.charAt(Math.min(CHARS.length - 1, idx));
      }
      lines.push(line);
    }
    el.textContent = lines.join("\n");
  }

  function tickAsciiSmoke() {
    if (!smokeEl || !smokeEl.isConnected) {
      stopAsciiSmoke();
      return;
    }

    var i, p;

    if (particles.length < MAX_PARTICLES && Math.random() < 0.8) {
      spawnParticle(true);
    }

    for (i = particles.length - 1; i >= 0; i--) {
      p = particles[i];
      p.wobble += 0.13;
      p.vx += Math.sin(p.wobble) * 0.007;
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.99;
      p.life -= p.decay;
      if (p.life <= 0 || p.y < -0.5) particles.splice(i, 1);
    }

    renderAsciiSmoke(smokeEl);
  }

  function startAsciiSmoke() {
    stopAsciiSmoke();
    if (prefersReducedMotion()) return;

    smokeEl = document.querySelector(".hero-smoke-ascii");
    if (!smokeEl) return;

    for (var i = 0; i < 36; i++) spawnParticle(false);
    renderAsciiSmoke(smokeEl);
    smokeTimer = setInterval(tickAsciiSmoke, TICK_MS);
  }

  var navigated = false;

  function onNavigate() {
    updateHeader();
    if (navigated) restartHeroEnter();
    navigated = true;
    startAsciiSmoke();
  }

  window.addEventListener("scroll", updateHeader, { passive: true });

  if (window.document$) {
    window.document$.subscribe(onNavigate);
  } else {
    document.addEventListener("DOMContentLoaded", onNavigate);
  }
})();
