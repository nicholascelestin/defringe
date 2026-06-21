// Interactive a*b* colour wheel that draws a pass's detection wedges live from their
// sliders, and writes them back when you drag the wedge edges. Vanilla web component.
//
// Each wedge exposes three drag handles, one per slider-backed edge:
//   • the two radial edges (hue floor / hue ceiling) — drag angularly
//   • the inner arc (min chroma)                      — drag radially
// Dragging dispatches `input` (live) then `pointerup` (release) on the target slider,
// the same path RESET_JS uses, so Gradio re-runs the pipeline on drag-end.
//
// config (set as a property): { maxChroma, wedges: [{ ref, color, chroma, lo, hi }] }
//   ref           reference hue in degrees (e.g. red=5, teal-green=200)
//   chroma/lo/hi  elem_ids of the min-chroma / hue-floor / hue-ceiling sliders
//   the wedge spans absolute hue [ref+lo, ref+hi], inner radius = chroma / maxChroma.
//
// The disc is the true CIELAB a*b* plane at a fixed lightness: angle = hue, radius =
// chroma, each pixel converted Lab -> sRGB. The wheel is built once and cached.
class DefringeWheel extends HTMLElement {
  connectedCallback() {
    if (this._ctx) return;
    this.config = this.config || {};
    this.size = 260;                                     // disc box (canvas height)
    this._margin = 16;                                   // disc rim -> top/bottom edge
    this.gutter = 64;                                    // side room for the Casters/Shadows captions
    this.w = this.size + 2 * this.gutter;                // canvas is wider than tall
    const cv = document.createElement('canvas');
    const dpr = window.devicePixelRatio || 1;
    cv.width = this.w * dpr; cv.height = this.size * dpr;
    cv.style.width = this.w + 'px';
    cv.style.maxWidth = '100%'; cv.style.height = 'auto';    // scale down (mobile) keeping aspect
    this.style.display = 'block';                        // fill the column...
    cv.style.display = 'block'; cv.style.margin = '0 auto';   // ...so the canvas centres
    this.appendChild(cv);
    this._ctx = cv.getContext('2d');
    this._dpr = dpr;
    this._ctx.scale(dpr, dpr);
    this._onInput = (e) => {
      if (this._drag) return;                           // our own writes redraw via the drag loop
      const b = e.target.closest && e.target.closest('[id^="def_"]');
      if (b && this._ids().indexOf(b.id) !== -1) this.draw();
    };
    document.addEventListener('input', this._onInput, true);
    this._handles = [];
    this._drag = null;
    this._hoverId = null;
    cv.style.touchAction = 'none';
    cv.addEventListener('pointerdown', (e) => this._down(e));
    cv.addEventListener('pointermove', (e) => this._move(e));
    cv.addEventListener('pointerup', (e) => this._up(e));
    cv.addEventListener('pointercancel', (e) => this._up(e));
    cv.addEventListener('pointerleave', () => this._leave());
    this._tries = 0;
    this._poll = setInterval(() => {                     // catch late/lazy slider render
      this.draw();
      if (++this._tries > 40) this._stopPoll();          // give up after ~12s
    }, 300);
    this.draw();
  }
  disconnectedCallback() {
    document.removeEventListener('input', this._onInput, true);
    this._stopPoll();
  }
  _stopPoll() { if (this._poll) { clearInterval(this._poll); this._poll = null; } }
  _geom() {                                             // disc centre/radius in drawing px
    return { cx: this.w / 2, cy: this.size / 2, R: this.size / 2 - this._margin };
  }
  // --- pointer interaction --------------------------------------------------
  _xy(e) {
    const r = this._ctx.canvas.getBoundingClientRect();  // map client px -> drawing px (canvas may be CSS-scaled)
    return [(e.clientX - r.left) * (this.w / r.width), (e.clientY - r.top) * (this.size / r.height)];
  }
  _hit(px, py) {
    let best = null, bd = 12 * 12;                       // 12px grab radius
    for (const h of this._handles) {
      const d = (h.x - px) * (h.x - px) + (h.y - py) * (h.y - py);
      if (d <= bd) { bd = d; best = h; }
    }
    return best;
  }
  _input(id) {                                          // the slider's range (or number) input
    return document.querySelector('#' + id + ' input[type=range], #' + id + ' input[type=number]');
  }
  _range(id) {
    const el = this._input(id);
    if (!el) return null;
    return { min: parseFloat(el.min), max: parseFloat(el.max), step: parseFloat(el.step) || 1 };
  }
  _setSlider(id, val) {
    const block = document.getElementById(id);
    if (!block) return;
    block.querySelectorAll('input').forEach((inp) => {
      inp.value = val;
      inp.dispatchEvent(new Event('input', { bubbles: true }));
    });
  }
  _releaseSlider(id) {                                   // mirror RESET_JS: pointerup triggers .release
    const block = document.getElementById(id);
    if (!block) return;
    block.querySelectorAll('input').forEach((inp) => {
      inp.dispatchEvent(new Event('pointerup', { bubbles: true }));
      inp.dispatchEvent(new Event('change', { bubbles: true }));
    });
  }
  _overReset(px, py) {
    const b = this._resetBtn;
    return b && px >= b.x0 && px <= b.x1 && py >= b.y0 && py <= b.y1;
  }
  _circArrow(ctx, x, y, r) {                            // counter-clockwise refresh arrow
    ctx.strokeStyle = '#f5f5f5'; ctx.fillStyle = '#f5f5f5'; ctx.lineWidth = 1.6; ctx.lineCap = 'round';
    const start = -0.25 * Math.PI, end = 1.25 * Math.PI; // 270° sweep, clear gap at top
    ctx.beginPath(); ctx.arc(x, y, r, start, end); ctx.stroke();
    const hx = x + r * Math.cos(end), hy = y + r * Math.sin(end), dir = end + Math.PI / 2;  // travel tangent
    const px = Math.cos(dir + Math.PI / 2), py = Math.sin(dir + Math.PI / 2);
    ctx.beginPath();
    ctx.moveTo(hx + 4 * Math.cos(dir), hy + 4 * Math.sin(dir));   // tip
    ctx.lineTo(hx + 3 * px, hy + 3 * py); ctx.lineTo(hx - 3 * px, hy - 3 * py);
    ctx.closePath(); ctx.fill();
  }
  _defaults() {                                         // read saved defaults fresh from the blob
    try {                                               // (window.__defaults can be a stale load-time snapshot)
      const t = document.querySelector('#defaults_blob textarea, #defaults_blob input');
      if (t && t.value) return JSON.parse(t.value);
    } catch (e) { /* fall through */ }
    return window.__defaults || {};
  }
  _reset() {                                            // restore the wheel's sliders to saved defaults
    const d = this._defaults();
    const ids = this._ids().filter((id) => d[id] !== undefined && document.getElementById(id));
    ids.forEach((id) => this._setSlider(id, d[id]));
    if (ids.length) this._releaseSlider(ids[ids.length - 1]);   // one pipeline re-run
    this.draw();
  }
  _down(e) {
    const [px, py] = this._xy(e);
    if (this._overReset(px, py)) { this._reset(); e.preventDefault(); return; }
    const h = this._hit(px, py);
    if (!h) return;
    this._drag = h;
    this._ctx.canvas.setPointerCapture(e.pointerId);
    this._ctx.canvas.style.cursor = 'grabbing';
    e.preventDefault();
  }
  _leave() {
    if (this._drag) return;
    this._ctx.canvas.style.cursor = 'default';
    let redraw = this._resetHot; this._resetHot = false;
    if (this._hoverId !== null) { this._hoverId = null; redraw = true; }
    if (redraw) this.draw();
  }
  _move(e) {
    if (!this._drag) {                                   // hover: reset button, then handles
      const [px, py] = this._xy(e);
      const onReset = this._overReset(px, py);
      const h = onReset ? null : this._hit(px, py);
      this._ctx.canvas.style.cursor = onReset ? 'pointer' : (h ? 'grab' : 'default');
      const id = h ? h.id : null;
      if (id !== this._hoverId || onReset !== this._resetHot) {
        this._hoverId = id; this._resetHot = onReset; this.draw();
      }
      return;
    }
    const h = this._drag, [px, py] = this._xy(e);
    const { cx, cy, R } = this._geom();
    const rg = this._range(h.id);
    if (!rg) return;
    let val;
    if (h.type === 'radius') {
      const r = Math.hypot(px - cx, py - cy);
      val = Math.max(0, Math.min(1, r / R)) * (this.config.maxChroma || 60);
    } else {                                             // angular edge -> hue offset from ref
      const deg = Math.atan2(-(py - cy), px - cx) * 180 / Math.PI;
      val = ((deg - h.ref + 180) % 360 + 360) % 360 - 180;
    }
    val = Math.round(val / rg.step) * rg.step;
    val = Math.max(rg.min, Math.min(rg.max, parseFloat(val.toFixed(4))));
    this._setSlider(h.id, val);
    this.draw();
    e.preventDefault();
  }
  _up(e) {
    if (!this._drag) return;
    const id = this._drag.id;
    this._drag = null;
    this._ctx.canvas.style.cursor = 'grab';
    this._releaseSlider(id);                             // fire .release -> pipeline re-runs
  }
  _ids() {
    const out = [];
    for (const w of (this.config.wedges || [])) out.push(w.chroma, w.lo, w.hi);
    return out;
  }
  _val(id) {
    const el = this._input(id);
    return el ? parseFloat(el.value) : null;
  }
  _rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return 'rgba(' + (n >> 16 & 255) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  _lab2rgb(L, a, b) {
    const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
    const f = (t) => (t > 6 / 29 ? t * t * t : 3 * (6 / 29) * (6 / 29) * (t - 4 / 29));
    const X = 0.95047 * f(fx), Y = f(fy), Z = 1.08883 * f(fz);            // D65
    const lin = [3.2406 * X - 1.5372 * Y - 0.4986 * Z,
                 -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
                 0.0557 * X - 0.2040 * Y + 1.0570 * Z];
    return lin.map((c) => {
      c = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(Math.max(0, c), 1 / 2.4) - 0.055;
      return Math.max(0, Math.min(1, c)) * 255;
    });
  }
  _buildWheel() {
    const px = Math.round(this.size * (this._dpr || 1)), c = px / 2;
    const R = (this.size / 2 - this._margin) * (this._dpr || 1);
    const maxC = this.config.maxChroma || 60, L = 74;
    const off = document.createElement('canvas');         // plain context, no transform
    off.width = off.height = px;
    const octx = off.getContext('2d');
    const img = octx.createImageData(px, px), d = img.data;
    for (let y = 0; y < px; y++) {
      for (let x = 0; x < px; x++) {
        const dx = x - c, dy = y - c, i = (y * px + x) * 4;
        if (dx * dx + dy * dy > R * R) { d[i + 3] = 0; continue; }
        const rgb = this._lab2rgb(L, maxC * dx / R, -maxC * dy / R);   // angle=hue, radius=chroma
        d[i] = rgb[0]; d[i + 1] = rgb[1]; d[i + 2] = rgb[2]; d[i + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    this._wheelCanvas = off;
  }
  draw() {
    const ctx = this._ctx, { cx, cy, R } = this._geom();
    const maxC = this.config.maxChroma || 60;
    if (!this._wheelCanvas) this._buildWheel();
    ctx.clearRect(0, 0, this.w, this.size);
    ctx.drawImage(this._wheelCanvas, cx - this.size / 2, 0, this.size, this.size);   // disc, centred
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(0,0,0,0.3)'; ctx.lineWidth = 1; ctx.stroke();
    const txt = getComputedStyle(this).color || '#888';
    // Side captions out in the gutters, clear of the disc: shadows near green (left),
    // casters near red (right). Centred in the space between the rim and the edge.
    ctx.font = 'bold 12px sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    ctx.fillStyle = txt;
    ctx.fillText('Shadows', (cx - R) / 2, cy);
    ctx.fillText('Casters', (cx + R + this.w) / 2, cy);
    ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI); ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fill();
    this._drawResetButton(ctx);
    const wedges = this.config.wedges || [];
    let resolved = 0;
    this._handles = [];
    for (const w of wedges) {
      const c = this._val(w.chroma), lo = this._val(w.lo), hi = this._val(w.hi);
      if (c == null || lo == null || hi == null) continue;   // slider not in DOM yet -> poll retries
      resolved++;
      const a0 = (w.ref + lo) * Math.PI / 180, a1 = (w.ref + hi) * Math.PI / 180;
      const rin = Math.max(0, Math.min(1, c / maxC)) * R;   // canvas y is down -> negate angle
      ctx.beginPath();
      ctx.arc(cx, cy, R, -a0, -a1, true);
      ctx.arc(cx, cy, rin, -a1, -a0, false);
      ctx.closePath();
      ctx.fillStyle = this._rgba(w.color, 0.12); ctx.fill();
      ctx.lineWidth = 2.5; ctx.strokeStyle = w.color; ctx.stroke();
      // handle anchors: mid-radius on each radial edge, inner-arc midpoint for chroma
      const rmid = (rin + R) / 2, amid = (a0 + a1) / 2;
      this._handles.push(
        { id: w.lo, type: 'angle', ref: w.ref, color: w.color, val: lo,
          x: cx + rmid * Math.cos(a0), y: cy - rmid * Math.sin(a0) },
        { id: w.hi, type: 'angle', ref: w.ref, color: w.color, val: hi,
          x: cx + rmid * Math.cos(a1), y: cy - rmid * Math.sin(a1) },
        { id: w.chroma, type: 'radius', ref: w.ref, color: w.color, val: c,
          x: cx + rin * Math.cos(amid), y: cy - rin * Math.sin(amid) });
    }
    const activeId = this._drag ? this._drag.id : this._hoverId;
    for (const h of this._handles) this._drawHandle(ctx, cx, cy, h, h.id === activeId);
    if (wedges.length && resolved === wedges.length) this._stopPoll();   // every wedge live -> stop catch-up
  }
  _drawResetButton(ctx) {                              // reset-to-defaults pill (icon + label), top-right
    ctx.font = 'bold 11px sans-serif'; ctx.textBaseline = 'middle';
    const label = 'Reset', tw = ctx.measureText(label).width;
    const ir = 6, padX = 8, gap = 5, h = 22, bw = padX + ir * 2 + gap + tw + padX;
    const x1 = this.w - 6, x0 = x1 - bw, y0 = 6, y1 = y0 + h;
    this._resetBtn = { x0, y0, x1, y1 };
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x0, y0, bw, h, h / 2);
    else ctx.rect(x0, y0, bw, h);
    ctx.fillStyle = this._resetHot ? 'rgba(20,20,20,0.85)' : 'rgba(20,20,20,0.6)'; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = this._resetHot ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.3)'; ctx.stroke();
    const icy = (y0 + y1) / 2;
    ctx.save(); this._circArrow(ctx, x0 + padX + ir, icy, ir); ctx.restore();
    ctx.fillStyle = '#f5f5f5'; ctx.textAlign = 'left';
    ctx.fillText(label, x0 + padX + ir * 2 + gap, icy + 0.5);
  }
  _drawHandle(ctx, cx, cy, h, expanded) {
    if (!expanded) {                                     // resting: a bold tick on the wedge line
      ctx.save();
      ctx.strokeStyle = h.color; ctx.lineWidth = 5; ctx.lineCap = 'round';
      ctx.beginPath();
      if (h.type === 'radius') {                         // arc segment on the inner edge
        const rin = Math.hypot(h.x - cx, h.y - cy);
        const amid = Math.atan2(-(h.y - cy), h.x - cx), dA = 6 / Math.max(rin, 1);
        ctx.arc(cx, cy, rin, -amid - dA, -amid + dA);
      } else {                                           // segment of the radial edge
        const a = (h.ref + h.val) * Math.PI / 180, ux = Math.cos(a), uy = -Math.sin(a);
        ctx.moveTo(h.x - ux * 6, h.y - uy * 6); ctx.lineTo(h.x + ux * 6, h.y + uy * 6);
      }
      ctx.stroke();
      ctx.restore();
    } else {
      // Grippable capsule along the handle's drag axis: tangential for the hue edges
      // (you slide along the arc), radial for chroma (you slide in/out).
      const a = (h.ref + h.val) * Math.PI / 180;
      const ax = h.type === 'radius' ? Math.atan2(h.y - cy, h.x - cx) : Math.atan2(Math.cos(a), Math.sin(a));
      const hl = 8, hw = 4.5;                            // capsule half-length / half-width
      ctx.save();
      ctx.translate(h.x, h.y); ctx.rotate(ax);
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(-hl, -hw, 2 * hl, 2 * hw, hw);
      else { ctx.arc(-hl + hw, 0, hw, Math.PI / 2, -Math.PI / 2); ctx.arc(hl - hw, 0, hw, -Math.PI / 2, Math.PI / 2); ctx.closePath(); }
      ctx.shadowColor = 'rgba(0,0,0,0.4)'; ctx.shadowBlur = 3; ctx.shadowOffsetY = 1;
      ctx.fillStyle = h.color; ctx.fill();
      ctx.shadowColor = 'transparent';
      ctx.lineWidth = 1.5; ctx.strokeStyle = '#fff'; ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1;   // grip ticks
      for (const gx of [-2.5, 0, 2.5]) { ctx.beginPath(); ctx.moveTo(gx, -2); ctx.lineTo(gx, 2); ctx.stroke(); }
      ctx.restore();
    }
    // value label — always shown, nudged radially outward so it clears the line/capsule
    const dx = h.x - cx, dy = h.y - cy, len = Math.hypot(dx, dy) || 1;
    const lx = h.x + (dx / len) * 16, ly = h.y + (dy / len) * 16;
    const valTxt = (Math.round(h.val * 10) / 10).toString();
    ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.strokeText(valTxt, lx, ly);
    ctx.fillStyle = h.color; ctx.fillText(valTxt, lx, ly);
  }
}
if (!customElements.get('defringe-wheel')) customElements.define('defringe-wheel', DefringeWheel);

// Bootstrap: upgrade each placeholder div Gradio renders into a wheel. A custom
// element inside gr.HTML can be sanitised away, so we mount from a plain div.
function mountDefringeWheels() {
  for (const div of document.querySelectorAll('.defringe-wheel-mount')) {
    if (div._mounted) continue;
    let cfg;
    try { cfg = JSON.parse(div.getAttribute('data-config') || '{}'); } catch (e) { continue; }
    div._mounted = true;
    const el = document.createElement('defringe-wheel');
    el.config = cfg;
    div.appendChild(el);
  }
}
new MutationObserver(mountDefringeWheels).observe(document.documentElement, { childList: true, subtree: true });
mountDefringeWheels();
