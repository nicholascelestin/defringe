// Gradio binding + load-time DOM glue: everything that knows about *this* app. The colour
// wheel (defringe_wheel.js) is host-agnostic; this file binds it to Gradio's sliders, mounts
// it into the placeholder divs gr.HTML renders, and wires the reset / collapsible / profile
// behaviours. Injected into <head> right after the wheel script.

// Mark the document JS-capable up front (before Gradio renders) so acc.css collapses the
// custom panels only when JS is present — with JS off they stay open and usable.
document.documentElement.classList.add('js');

// Saved per-slider defaults, read FRESH from the hidden #defaults_blob textbox each time
// (window.__defaults can be a stale load-time snapshot).
function readDefaults() {
  try {
    const t = document.querySelector('#defaults_blob textarea, #defaults_blob input');
    if (t) return JSON.parse(t.value || '{}');
  } catch (e) { /* fall through */ }
  return window.__defaults || {};
}

// The wheel's binding to a Gradio Slider: read its <input>; write live via `input`; commit
// via `pointerup` (what Gradio's .release listens for) + `change`. Defaults come from the
// app's hidden #defaults_blob. Same shape as the wheel's built-in DOM_ADAPTER, but commit
// fires the event Gradio re-runs the pipeline on.
const GRADIO_ADAPTER = {
  read(id) {
    const el = document.querySelector('#' + id + ' input[type=range], #' + id + ' input[type=number]');
    if (!el) return null;
    return { value: parseFloat(el.value), min: parseFloat(el.min), max: parseFloat(el.max), step: parseFloat(el.step) || 1 };
  },
  write(id, value) {
    const el = document.getElementById(id);
    if (el) el.querySelectorAll('input').forEach((inp) => { inp.value = value; inp.dispatchEvent(new Event('input', { bubbles: true })); });
  },
  commit(id) {
    const el = document.getElementById(id);
    if (el) el.querySelectorAll('input').forEach((inp) => {
      inp.dispatchEvent(new Event('pointerup', { bubbles: true }));   // Gradio Slider .release
      inp.dispatchEvent(new Event('change', { bubbles: true }));
    });
  },
  defaults() { return readDefaults(); },
};

// Mount each placeholder div gr.HTML renders into a wheel bound to the Gradio adapter. A
// custom element written directly into gr.HTML can be sanitised away, so we mount from a
// plain div carrying the spec in data-config.
function mountDefringeWheels() {
  for (const div of document.querySelectorAll('.defringe-wheel-mount')) {
    if (div._mounted) continue;
    let cfg;
    try { cfg = JSON.parse(div.getAttribute('data-config') || '{}'); } catch (e) { continue; }
    div._mounted = true;
    const el = document.createElement('defringe-wheel');
    el.config = cfg;
    el.adapter = GRADIO_ADAPTER;                         // bind to Gradio before connectedCallback
    div.appendChild(el);
  }
}
new MutationObserver(mountDefringeWheels).observe(document.documentElement, { childList: true, subtree: true });
mountDefringeWheels();

// (1) Per-slider reset (↺): override Gradio's native reset (which restores the build-time
// value) so it restores the active *profile* default instead, then re-runs the pipeline.
// Capture phase + stopImmediatePropagation so we beat Gradio's own handler to the click.
window.installResetHijack = function () {
  window.__defaults = readDefaults();
  if (window.__resetHijack) return;
  window.__resetHijack = true;
  document.addEventListener('click', (e) => {
    const btn = e.target.closest && e.target.closest('.reset-button');
    if (!btn) return;
    const block = btn.closest('[id^="def_"]');
    if (!block) return;
    const d = readDefaults()[block.id];
    if (d === undefined) return;                 // unknown -> let native reset run
    e.preventDefault(); e.stopImmediatePropagation();
    block.querySelectorAll('input').forEach((inp) => {
      inp.value = d;
      inp.dispatchEvent(new Event('input', { bubbles: true }));
      inp.dispatchEvent(new Event('pointerup', { bubbles: true }));
    });
  }, true);
};

// (2) Custom collapsibles: toggle `.open` on a header and its body (delegated, so it works
// for panels that mount later on a tab switch). acc.css hides bodies by default (with JS on).
window.installAccordion = function () {
  if (window.__accWired) return;
  window.__accWired = true;
  document.addEventListener('click', (e) => {
    const h = e.target.closest('.acc-head');
    if (!h) return;
    const body = document.getElementById(h.getAttribute('data-target'));
    if (!body) return;
    h.classList.toggle('open', body.classList.toggle('open'));
  });
};

// (3) After a profile load, the server hands us the fresh {elem_id: value} blob; resync the
// reset defaults and repaint every wheel.
window.applyProfileDefaults = function (b) {
  window.__defaults = JSON.parse(b || '{}');
  document.querySelectorAll('defringe-wheel').forEach((w) => w.draw && w.draw());
};
