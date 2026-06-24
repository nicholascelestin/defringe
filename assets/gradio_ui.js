// Host glue for the colour wheel: the controlled view (defringe_wheel.js) emits changes by its own
// opaque param KEY and never names a slider — the key -> slider-elem-id map (from views.py) plus ALL
// slider read/write/commit/snapshot live here. Classic inline <script> (DOM + window.* only), injected
// right after the wheel module.

// Active per-slider defaults, read FRESH from #defaults_blob each call (window.__defaults can be stale).
function readDefaults() {
  try {
    const blob = document.querySelector('#defaults_blob textarea, #defaults_blob input');
    if (blob) return JSON.parse(blob.value || '{}');
  } catch (e) { /* fall through */ }
  return window.__defaults || {};
}

// ── one home for "read / write / commit a Gradio slider" (each has both a range and a number input) ──
function readSlider(id) {
  const block = document.getElementById(id);
  // prefer the range input: it holds the live drag value before Gradio syncs the number input
  const input = block && (block.querySelector('input[type=range]') || block.querySelector('input[type=number]'));
  if (!input) return null;
  return { value: parseFloat(input.value), min: parseFloat(input.min), max: parseFloat(input.max), step: parseFloat(input.step) || 1 };
}
function eachInput(id, fn) {
  const block = document.getElementById(id);
  if (block) block.querySelectorAll('input').forEach(fn);
}
function writeSlider(id, value) {
  eachInput(id, (input) => { input.value = value; input.dispatchEvent(new Event('input', { bubbles: true })); });
}
function commitSlider(id) {
  eachInput(id, (input) => {
    input.dispatchEvent(new Event('pointerup', { bubbles: true }));   // Gradio Slider .release
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

// Bind a wheel to its sliders; returns snapshot() so a profile load can resync.
function bindWheel(el, sliderFor) {                        // sliderFor: wheel param key -> slider elem_id
  const keys = Object.keys(sliderFor);
  const sliderIds = Object.values(sliderFor);
  const ownedSliders = sliderIds.map((id) => '#' + id).join(',');   // selector for the wheel's slider blocks
  let applying = false;                                   // set only while WE write a slider, so our writes skip the input watcher

  // No value-dedup: the wheel local-echoes drags, so a snapshot matching a stale cache must still
  // repaint. Drag churn is held off by `applying` + the presence observer instead.
  function snapshot() {
    const values = {};
    for (const key of keys) { const reading = readSlider(sliderFor[key]); if (reading) values[key] = reading; }
    el.values = values;
  }

  function applyToSliders(write) { applying = true; try { write(); } finally { applying = false; } }

  el.addEventListener('paraminput', (e) => applyToSliders(() => writeSlider(sliderFor[e.detail.id], e.detail.value)));
  el.addEventListener('paramcommit', (e) => applyToSliders(() => commitSlider(sliderFor[e.detail.id])));
  el.addEventListener('paramreset', () => applyToSliders(() => {
    const defaults = readDefaults();
    const resettable = sliderIds.filter((id) => defaults[id] !== undefined && document.getElementById(id));
    resettable.forEach((id) => writeSlider(id, defaults[id]));
    if (resettable.length) commitSlider(resettable[resettable.length - 1]);   // one pipeline re-run
    snapshot();                                           // reset bypasses the wheel's local echo
  }));

  // External slider changes (drag, ↺ reset, etc.) re-snapshot -> repaint; our own writes are skipped
  // via `applying`. Match the slider BLOCK, not closest('[id]') — Gradio's range input has its own id.
  document.addEventListener('input', (e) => {
    if (applying) return;
    if (ownedSliders && e.target.closest && e.target.closest(ownedSliders)) snapshot();
  }, true);

  // Re-snapshot when the SET of present sliders changes (late mount, tab detach/reattach). Keyed on
  // presence, never value, so a wheel-driven DOM write can't trip it.
  const presentIds = () => sliderIds.filter((id) => document.getElementById(id)).join(',');
  let present = presentIds();
  new MutationObserver(() => {
    const resolved = presentIds();
    if (resolved === present) return;
    present = resolved;
    snapshot();
  }).observe(document.documentElement, { childList: true, subtree: true });

  snapshot();
  return snapshot;
}

// gr.HTML sanitises a custom element written into it, so mount each wheel from a plain placeholder div.
const wheelSnapshots = new Map();                         // el -> its snapshot(), for profile-load resync
function mountDefringeWheels() {
  for (const div of document.querySelectorAll('.defringe-wheel-mount')) {
    if (div._mounted) continue;
    let cfg, sliderFor;
    try {
      cfg = JSON.parse(div.getAttribute('data-config') || '{}');
      sliderFor = JSON.parse(div.getAttribute('data-param-map') || '{}');   // wheel param key -> slider elem_id (views.py)
    } catch (e) { continue; }
    div._mounted = true;
    const el = document.createElement('defringe-wheel');
    div.appendChild(el);
    // Set config/values only AFTER upgrade: they're accessors, and a pre-upgrade assignment becomes an
    // own property that shadows the setter (gradio_ui is classic; the wheel module is deferred).
    customElements.whenDefined('defringe-wheel').then(() => {
      el.config = cfg;
      wheelSnapshots.set(el, bindWheel(el, sliderFor));
    });
  }
}
new MutationObserver(mountDefringeWheels).observe(document.documentElement, { childList: true, subtree: true });
mountDefringeWheels();

// Per-slider ↺: restore the active PROFILE default (not Gradio's build-time value), then re-run.
// Capture + stopImmediatePropagation to beat Gradio's own click handler.
window.installResetHijack = function () {
  window.__defaults = readDefaults();
  if (window.__resetHijack) return;
  window.__resetHijack = true;
  document.addEventListener('click', (e) => {
    const btn = e.target.closest && e.target.closest('.reset-button');
    if (!btn) return;
    const block = btn.closest('[id^="def_"]');
    if (!block) return;
    const value = readDefaults()[block.id];
    if (value === undefined) return;                 // unknown -> let native reset run
    e.preventDefault(); e.stopImmediatePropagation();
    eachInput(block.id, (input) => {
      input.value = value;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('pointerup', { bubbles: true }));
    });
  }, true);
};

// Custom collapsibles: toggle `.open` on a header + its body (delegated, survives tab-switch remounts).
window.installAccordion = function () {
  if (window.__accWired) return;
  window.__accWired = true;
  document.addEventListener('click', (e) => {
    const head = e.target.closest('.acc-head');
    if (!head) return;
    const body = document.getElementById(head.getAttribute('data-target'));
    if (!body) return;
    head.classList.toggle('open', body.classList.toggle('open'));
  });
};

// After a profile load: refresh defaults from the server blob, then re-snapshot every wheel.
window.applyProfileDefaults = function (blob) {
  window.__defaults = JSON.parse(blob || '{}');
  wheelSnapshots.forEach((snapshot) => snapshot());
};
