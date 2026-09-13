// MathJax, loaded only on pages that actually contain math. The library is
// ~1 MB, and only one page uses it, so pulling it in from extra_javascript
// made every other page (including the landing page) wait on it.
//
// pymdownx.arithmatex with `generic: true` wraps every expression in
// .arithmatex, so that class is a reliable per-page signal. The check runs on
// document$ rather than once at startup, because Material's instant navigation
// swaps pages without a full document load.

var MATHJAX_SRC = "https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js";

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

(function () {
  "use strict";

  var loading = null;

  function loadMathJax() {
    if (loading) return loading;

    loading = new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = MATHJAX_SRC;
      script.async = true;
      script.onload = resolve;
      script.onerror = function () {
        // Allow a later page to retry rather than staying permanently broken.
        loading = null;
        reject(new Error("failed to load MathJax"));
      };
      document.head.appendChild(script);
    });

    return loading;
  }

  function typeset() {
    if (!document.querySelector(".arithmatex")) return;

    loadMathJax()
      .then(function () {
        return window.MathJax.typesetPromise();
      })
      .catch(function () {
        /* leave the raw TeX visible rather than blanking the page */
      });
  }

  if (window.document$) {
    window.document$.subscribe(typeset);
  } else {
    document.addEventListener("DOMContentLoaded", typeset);
  }
})();
