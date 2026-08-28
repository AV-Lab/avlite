---
hide:
  - navigation
  - toc
---

<div class="plugin-store" markdown>

# Community Plugins

<div class="store-hero" markdown>

## Plugin ecosystem

Community-built bridges, controllers, planners, and predictors for AVLite.
The list below is loaded live from the
[community plugin registry](https://github.com/AV-Lab/avlite-community-plugins),
with stats straight from GitHub.

<div class="store-counters" id="store-counters" aria-live="polite">
  <div class="store-counter"><span class="store-counter-value" id="counter-plugins">–</span><span class="store-counter-label">Plugins</span></div>
  <div class="store-counter"><span class="store-counter-value" id="counter-categories">–</span><span class="store-counter-label">Categories</span></div>
</div>

<div class="store-hero-actions" markdown>
[Submit your plugin](plugin-development.md#11-publish-to-the-community-registry-pull-request){ .md-button .md-button--primary }
[Build a plugin](plugin-development.md){ .md-button }
[Support](support.md){ .md-button }
</div>

</div>

<div class="store-toolbar" id="store-toolbar" hidden>
  <input class="store-search" id="store-search" type="search"
         placeholder="Search plugins…" aria-label="Search plugins">
  <div class="store-chips" id="store-chips" role="group" aria-label="Filter by category"></div>
  <label class="store-sort-label">Sort
    <select class="store-sort" id="store-sort" aria-label="Sort plugins">
      <option value="stars">Stars</option>
      <option value="updated">Recently updated</option>
      <option value="name">Name</option>
    </select>
  </label>
</div>

<div class="store-grid" id="store-grid" aria-live="polite">
  <!-- Populated by javascripts/community.js; skeleton cards shown while loading. -->
</div>

<div class="store-fallback" id="store-fallback" hidden>
  Couldn't load the plugin registry right now. Browse it directly on
  <a href="https://github.com/AV-Lab/avlite-community-plugins">GitHub</a>.
</div>

<noscript>
  <div class="store-fallback">
    This page needs JavaScript to list plugins. Browse the registry directly on
    <a href="https://github.com/AV-Lab/avlite-community-plugins">GitHub</a>.
  </div>
</noscript>

<div class="store-install" markdown>

### Installing plugins

Plugins are installed from inside AVLite — open the Plugins browser, pick a
plugin from the **Community** tab, and click install:

```bash
avlite plugins
```

Want to write your own? Start with the
[plugin development guide](plugin-development.md).

</div>

</div>
