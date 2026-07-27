// Community Plugins page ("app store"): fetches the live plugin registry
// (plugins.yaml) and public GitHub repo stats, then renders a searchable,
// filterable, sortable card grid. Loaded site-wide via extra_javascript, so it
// only acts when the page contains #store-grid, and re-initializes on
// Material's instant navigation via the document$ observable. Its js-yaml
// dependency is fetched on demand so other pages never download it.
(function () {
  "use strict";

  var REGISTRY_URL =
    "https://raw.githubusercontent.com/AV-Lab/avlite-community-plugins/main/plugins.yaml";
  var JS_YAML_URL =
    "https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js";
  var CACHE_PREFIX = "avlite-store:";
  var CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

  // ---------------------------------------------------------------- caching

  function cacheGet(key) {
    try {
      var raw = localStorage.getItem(CACHE_PREFIX + key);
      if (!raw) return null;
      var entry = JSON.parse(raw);
      if (Date.now() - entry.t > CACHE_TTL_MS) return null;
      return entry.v;
    } catch (e) {
      return null;
    }
  }

  function cacheSet(key, value) {
    try {
      localStorage.setItem(
        CACHE_PREFIX + key,
        JSON.stringify({ t: Date.now(), v: value })
      );
    } catch (e) {
      /* storage full or disabled — live without the cache */
    }
  }

  // ----------------------------------------------------------------- fetch

  var yamlLoading = null;

  // Fetched here rather than site-wide, so the other pages never pay for it.
  // A warm registry cache skips the parser entirely.
  function loadJsYaml() {
    if (window.jsyaml) return Promise.resolve();
    if (yamlLoading) return yamlLoading;

    yamlLoading = new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = JS_YAML_URL;
      script.async = true;
      script.onload = resolve;
      script.onerror = function () {
        yamlLoading = null;
        reject(new Error("failed to load js-yaml"));
      };
      document.head.appendChild(script);
    });

    return yamlLoading;
  }

  function fetchRegistry() {
    var cached = cacheGet("registry");
    if (cached) return Promise.resolve(cached);
    return Promise.all([fetch(REGISTRY_URL), loadJsYaml()])
      .then(function (results) {
        var res = results[0];
        if (!res.ok) throw new Error("registry HTTP " + res.status);
        return res.text();
      })
      .then(function (text) {
        var doc = window.jsyaml.load(text);
        var plugins = (doc && doc.plugins) || [];
        cacheSet("registry", plugins);
        return plugins;
      });
  }

  // Returns {stars, forks, issues, pushed_at} or null (rate-limited/offline).
  function fetchRepoStats(repoUrl) {
    var m = /github\.com\/([^\/]+)\/([^\/#?]+)/.exec(repoUrl || "");
    if (!m) return Promise.resolve(null);
    var slug = m[1] + "/" + m[2].replace(/\.git$/, "");
    var cached = cacheGet("repo:" + slug);
    if (cached) return Promise.resolve(cached);
    return fetch("https://api.github.com/repos/" + slug)
      .then(function (res) {
        if (!res.ok) throw new Error("api HTTP " + res.status);
        return res.json();
      })
      .then(function (data) {
        var stats = {
          stars: data.stargazers_count || 0,
          forks: data.forks_count || 0,
          issues: data.open_issues_count || 0,
          pushed_at: data.pushed_at || null,
        };
        cacheSet("repo:" + slug, stats);
        return stats;
      })
      .catch(function () {
        return null;
      });
  }

  // ------------------------------------------------------------- utilities

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[c];
    });
  }

  // "WorldBridge" -> "World Bridge", "PredictionStrategy" -> "Prediction"
  function categoryLabel(cat) {
    return String(cat)
      .replace(/Strategy$/, "")
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2");
  }

  function relativeTime(iso) {
    if (!iso) return null;
    var s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 0) s = 0;
    var units = [
      [31536000, "year"],
      [2592000, "month"],
      [604800, "week"],
      [86400, "day"],
      [3600, "hour"],
      [60, "minute"],
    ];
    for (var i = 0; i < units.length; i++) {
      var n = Math.floor(s / units[i][0]);
      if (n >= 1) return n + " " + units[i][1] + (n > 1 ? "s" : "") + " ago";
    }
    return "just now";
  }

  function svgIcon(name) {
    var paths = {
      star: "M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.75.75 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z",
      fork: "M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-.878a2.25 2.25 0 1 1 1.5 0v.878a2.25 2.25 0 0 1-2.25 2.25h-1.5v2.128a2.251 2.251 0 1 1-1.5 0V8.5h-1.5A2.25 2.25 0 0 1 3.5 6.25v-.878a2.25 2.25 0 1 1 1.5 0ZM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Zm6.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm-3 8.75a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z",
      issue:
        "M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z",
      clock:
        "M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm7-3.25v3.06l2.03 2.03a.75.75 0 1 1-1.06 1.06L7.22 8.65a.75.75 0 0 1-.22-.53V4.75a.75.75 0 0 1 1.5 0Z",
    };
    return (
      '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
      '<path fill="currentColor" d="' +
      paths[name] +
      '"/></svg>'
    );
  }

  function formatCount(n) {
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    return String(n);
  }

  // -------------------------------------------------------------- rendering

  function skeletonCards(count) {
    var html = "";
    for (var i = 0; i < count; i++) {
      html +=
        '<div class="store-card store-card--skeleton" aria-hidden="true">' +
        '<div class="store-skel store-skel--title"></div>' +
        '<div class="store-skel store-skel--line"></div>' +
        '<div class="store-skel store-skel--line short"></div>' +
        '<div class="store-skel store-skel--chip"></div>' +
        "</div>";
    }
    return html;
  }

  function renderCard(p) {
    var stats = p._stats;
    var cats = Array.isArray(p.category) ? p.category : [p.category];
    var author = p.author || "";

    var statsHtml = "";
    if (stats) {
      var updated = relativeTime(stats.pushed_at);
      statsHtml =
        '<div class="store-card-stats">' +
        '<span title="Stars">' + svgIcon("star") + formatCount(stats.stars) + "</span>" +
        '<span title="Forks">' + svgIcon("fork") + formatCount(stats.forks) + "</span>" +
        '<span title="Open issues">' + svgIcon("issue") + formatCount(stats.issues) + "</span>" +
        (updated
          ? '<span title="Last push">' + svgIcon("clock") + escapeHtml(updated) + "</span>"
          : "") +
        "</div>";
    }

    var notes = p.dependency_notes
      ? '<p class="store-card-notes">' + escapeHtml(p.dependency_notes) + "</p>"
      : "";

    var minVer = p.min_avlite_version
      ? '<span class="store-chip store-chip--version">avlite ≥ ' +
        escapeHtml(p.min_avlite_version) +
        "</span>"
      : "";

    return (
      '<a class="store-card" href="' +
      escapeHtml(p.repository) +
      '" target="_blank" rel="noopener">' +
      '<div class="store-card-head">' +
      '<img class="store-card-avatar" src="https://github.com/' +
      encodeURIComponent(author) +
      '.png?size=64" alt="" loading="lazy" onerror="this.style.display=\'none\'">' +
      "<div>" +
      '<span class="store-card-name">' + escapeHtml(p.name) + "</span>" +
      '<span class="store-card-author">by ' + escapeHtml(author) + "</span>" +
      "</div>" +
      "</div>" +
      '<p class="store-card-desc">' + escapeHtml(p.description) + "</p>" +
      '<div class="store-card-tags">' +
      cats
        .map(function (c) {
          return '<span class="store-chip">' + escapeHtml(categoryLabel(c)) + "</span>";
        })
        .join("") +
      minVer +
      "</div>" +
      notes +
      statsHtml +
      "</a>"
    );
  }

  // ------------------------------------------------------------------ init

  function init() {
    var grid = document.getElementById("store-grid");
    if (!grid || grid.dataset.storeInit) return;
    grid.dataset.storeInit = "1";

    var toolbar = document.getElementById("store-toolbar");
    var searchBox = document.getElementById("store-search");
    var chipBox = document.getElementById("store-chips");
    var sortBox = document.getElementById("store-sort");
    var fallback = document.getElementById("store-fallback");

    grid.innerHTML = skeletonCards(6);

    var plugins = [];
    var activeCategory = null;

    function applyView() {
      var q = (searchBox.value || "").trim().toLowerCase();
      var shown = plugins.filter(function (p) {
        var cats = Array.isArray(p.category) ? p.category : [p.category];
        if (activeCategory && cats.indexOf(activeCategory) === -1) return false;
        if (!q) return true;
        return (
          (p.name || "").toLowerCase().indexOf(q) !== -1 ||
          (p.description || "").toLowerCase().indexOf(q) !== -1 ||
          (p.author || "").toLowerCase().indexOf(q) !== -1
        );
      });

      var sort = sortBox.value;
      shown.sort(function (a, b) {
        var sa = a._stats, sb = b._stats;
        if (sort === "stars") {
          return (sb ? sb.stars : -1) - (sa ? sa.stars : -1);
        }
        if (sort === "updated") {
          var ta = sa && sa.pushed_at ? new Date(sa.pushed_at).getTime() : 0;
          var tb = sb && sb.pushed_at ? new Date(sb.pushed_at).getTime() : 0;
          return tb - ta;
        }
        return (a.name || "").localeCompare(b.name || "");
      });

      grid.innerHTML = shown.length
        ? shown.map(renderCard).join("")
        : '<div class="store-empty">No plugins match your search.</div>';
    }

    function buildChips() {
      var cats = [];
      plugins.forEach(function (p) {
        (Array.isArray(p.category) ? p.category : [p.category]).forEach(
          function (c) {
            if (c && cats.indexOf(c) === -1) cats.push(c);
          }
        );
      });
      cats.sort();

      var html =
        '<button class="store-chip store-chip--filter is-active" data-cat="">All</button>';
      cats.forEach(function (c) {
        html +=
          '<button class="store-chip store-chip--filter" data-cat="' +
          escapeHtml(c) +
          '">' +
          escapeHtml(categoryLabel(c)) +
          "</button>";
      });
      chipBox.innerHTML = html;

      chipBox.addEventListener("click", function (ev) {
        var btn = ev.target.closest("button[data-cat]");
        if (!btn) return;
        activeCategory = btn.dataset.cat || null;
        chipBox.querySelectorAll("button").forEach(function (b) {
          b.classList.toggle("is-active", b === btn);
        });
        applyView();
      });

      return cats;
    }

    function updateCounters(cats) {
      var el;
      if ((el = document.getElementById("counter-plugins")))
        el.textContent = String(plugins.length);
      if ((el = document.getElementById("counter-categories")))
        el.textContent = String(cats.length);
    }

    fetchRegistry()
      .then(function (list) {
        plugins = list;
        var cats = buildChips();
        toolbar.hidden = false;

        // First paint without stats, then enrich as stats arrive.
        applyView();
        updateCounters(cats);

        return Promise.all(
          plugins.map(function (p) {
            return fetchRepoStats(p.repository).then(function (stats) {
              p._stats = stats;
            });
          })
        ).then(function () {
          applyView();
          updateCounters(cats);
        });
      })
      .catch(function (err) {
        console.warn("community plugins: failed to load registry", err);
        grid.innerHTML = "";
        if (fallback) fallback.hidden = false;
      });

    searchBox.addEventListener("input", applyView);
    sortBox.addEventListener("change", applyView);
  }

  if (window.document$) {
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
