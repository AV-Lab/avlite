// Landing-page header treatment: transparent/blurred over the hero at the top,
// solid (primary) once scrolled or on any inner page. Works with Material's
// instant navigation via the document$ observable.
(function () {
  function update() {
    var isLanding = !!document.querySelector(".hero");
    document.body.classList.toggle("landing-home", isLanding);
    document.body.classList.toggle(
      "landing-scrolled",
      isLanding && window.scrollY > 40
    );
  }

  window.addEventListener("scroll", update, { passive: true });

  if (window.document$) {
    window.document$.subscribe(update);
  } else {
    document.addEventListener("DOMContentLoaded", update);
  }
})();
