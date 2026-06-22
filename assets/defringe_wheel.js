// Interactive a*b* colour wheel: draws a pass's detection wedges live from a set of bound
// numeric parameters, and writes them back when you drag the wedge edges. Framework-agnostic
// vanilla web component — it touches its parameters ONLY through an injected `adapter`, so it
// has no knowledge of Gradio (or any host). See gradio_ui.js for the Gradio binding + mount.
//
// Properties (set before the element is appended):
//   config   { maxChroma, wedges:[{ ref, color, chroma, lo, hi }], sizeReach, repair, ... }
//            ref = reference hue in degrees (e.g. red=5, teal-green=200); chroma/lo/hi = opaque
//            parameter ids; the wedge spans absolute hue [ref+lo, ref+hi], inner r = chroma/maxChroma.
//   adapter  binding to the host's controls (defaults to DOM_ADAPTER below):
//            read(id)->{value,min,max,step}|null · write(id,v) · commit(id) · defaults()->{id:value}
//
// Each wedge exposes three drag handles (the hue-floor/ceiling edges drag angularly, the inner
// arc drags radially). Dragging calls adapter.write (live) then adapter.commit (on release).
// The disc is the true CIELAB a*b* plane at a fixed lightness; built once and cached.

// Default binding: each id addresses a DOM container `#id` holding a range/number <input>.
// Read live; writes fire `input`; commit fires `change`. Replace via `el.adapter = …` to bind
// to another host (see gradio_ui.js, which fires `pointerup` for Gradio's .release).
const DOM_ADAPTER = {
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
    if (el) el.querySelectorAll('input').forEach((inp) => inp.dispatchEvent(new Event('change', { bubbles: true })));
  },
  defaults() { return window.__wheelDefaults || {}; },
};

class DefringeWheel extends HTMLElement {
  connectedCallback() {
    if (this._ctx) return;
    this.config = this.config || {};
    this.adapter = this.adapter || DOM_ADAPTER;          // host binding; DOM by default
    this.size = 260;                                     // disc box (square)
    this._margin = 16;                                   // disc rim -> top/bottom edge
    this.gutter = 64;                                    // side room for the Casters/Shadows captions
    this.foot = (this.config.sizeReach || this.config.repair) ? 132 : 0;   // bottom strip for the blobs
    this.padLeft = (this.config.layout === 'source') ? 64 : 0;   // extra left margin so the caster blob + its labels clear the colour band
    this.w = this.size + 2 * this.gutter + this.padLeft;   // canvas is wider than tall
    this.h = this.size + this.foot;                      // total canvas height
    const cv = document.createElement('canvas');
    const dpr = window.devicePixelRatio || 1;
    cv.width = this.w * dpr; cv.height = this.h * dpr;
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
      const b = e.target.closest && e.target.closest('[id]');
      if (b && this._ids().indexOf(b.id) !== -1) this.draw();   // only my own bound controls
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
    // The host may render the bound controls after the wheel mounts (and gives no "ready"
    // signal), so watch the DOM and redraw as they resolve (rAF-coalesced); draw() disconnects
    // us once every wedge reads non-null.
    this._resolveObs = new MutationObserver(() => {
      if (this._rafPending) return;
      this._rafPending = true;
      requestAnimationFrame(() => { this._rafPending = false; this.draw(); });
    });
    this._resolveObs.observe(document.documentElement, { childList: true, subtree: true });
    this.draw();
  }
  disconnectedCallback() {
    document.removeEventListener('input', this._onInput, true);
    this._stopResolve();
  }
  _stopResolve() { if (this._resolveObs) { this._resolveObs.disconnect(); this._resolveObs = null; } }
  _geom() {                                             // disc centre/radius in drawing px
    return { cx: this.size / 2 + this.gutter + (this.padLeft || 0), cy: this.size / 2, R: this.size / 2 - this._margin };
  }
  // --- pointer interaction --------------------------------------------------
  _xy(e) {
    const r = this._ctx.canvas.getBoundingClientRect();  // map client px -> drawing px (canvas may be CSS-scaled)
    return [(e.clientX - r.left) * (this.w / r.width), (e.clientY - r.top) * (this.h / r.height)];
  }
  _hit(px, py) {
    let best = null, bd = 12 * 12;                       // 12px grab radius
    for (const h of this._handles) {
      const d = (h.x - px) * (h.x - px) + (h.y - py) * (h.y - py);
      if (d <= bd) { bd = d; best = h; }
    }
    return best;
  }
  _range(id) {
    const r = this.adapter.read(id);
    return r ? { min: r.min, max: r.max, step: r.step } : null;
  }
  // value <-> radius for the foot-blob knobs. Concave (gamma < 1) so low/default values get
  // generous room and a large slider range still fits the blob's designated radius -- i.e. the
  // knobs read big by default and compress as you drag toward the edge. _shrink inverts _grow.
  _grow(v, vmax, Lmax) { return Lmax * Math.pow(Math.max(0, v) / Math.max(vmax, 1e-6), 0.6); }
  _shrink(px, vmax, Lmax) { return Math.max(vmax, 1e-6) * Math.pow(Math.max(0, px) / Math.max(Lmax, 1e-6), 1 / 0.6); }
  _setSlider(id, val) { this.adapter.write(id, val); }   // live update through the binding
  _releaseSlider(id) { this.adapter.commit(id); }        // finalize -> host re-runs its pipeline
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
  _defaults() { return this.adapter.defaults(); }        // reset targets (host-provided)
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
    if (h.type === 'blob') {                              // blob radius -> Minimum Area
      val = this._shrink(Math.hypot(px - h.bcx, py - h.bcy), h.vmax, h.lmax);
    } else if (h.type === 'reach') {                      // ring gap beyond blob -> Cast Reach
      val = this._shrink(Math.max(0, Math.hypot(px - h.bcx, py - h.bcy) - h.blobR), h.vmax, h.lmax);
    } else if (h.type === 'strength') {                   // vertical position in the blob -> Maximum Strength
      val = (h.cy + h.R0 - py) / (2 * h.R0);
    } else if (h.type === 'spread') {                     // grown radius beyond base blob -> Repair Spread
      val = this._shrink(Math.max(0, Math.hypot(px - h.bcx, py - h.bcy) - h.R0), h.vmax, h.lmax);
    } else if (h.type === 'feather') {                    // ring beyond spread blob -> Feather
      val = this._shrink(Math.max(0, Math.hypot(px - h.bcx, py - h.bcy) - h.Rs), h.vmax, h.lmax);
    } else if (h.type === 'plight') {                     // vertical level in the source blob -> Min Lightness
      const top = h.bcy - h.blobR, bot = h.bcy + h.blobR;
      val = rg.max - ((py - top) / Math.max(bot - top, 1)) * (rg.max - rg.min);
    } else if (h.type === 'ring') {                       // ring radius -> Excess Threshold (radial distance from ambient)
      val = this._shrink(Math.hypot(px - h.bcx, py - h.bcy), h.vmax, h.lmax);
    } else if (h.type === 'radius') {
      const r = Math.hypot(px - cx, py - cy);
      val = Math.max(0, Math.min(1, r / R)) * (this.config.maxChroma || 60);
    } else if (h.type === 'halfwidth') {                 // edge -> angular distance from the axis
      const deg = Math.atan2(-(py - cy), px - cx) * 180 / Math.PI;
      val = Math.abs(((deg - (h.ref + h.center + (h.doff || 0)) + 180) % 360 + 360) % 360 - 180);
    } else {                                             // angular edge -> hue offset from ref (minus any display rotation)
      const deg = Math.atan2(-(py - cy), px - cx) * 180 / Math.PI;
      val = ((deg - h.ref - (h.doff || 0) + 180) % 360 + 360) % 360 - 180;
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
    for (const w of (this.config.wedges || []))
      for (const k of ['chroma', 'lo', 'hi', 'center', 'halfwidth']) if (w[k]) out.push(w[k]);
    const sr = this.config.sizeReach;
    if (sr) out.push(sr.area, sr.reach);
    const src = this.config.source;
    if (src) out.push(src.lightness, src.area, src.reach);
    const er = this.config.excessRing;
    if (er) out.push(er.threshold);
    const rp = this.config.repair;
    if (rp) out.push(rp.strength, rp.spread, rp.feather);
    return out;
  }
  _val(id) {
    const r = this.adapter.read(id);
    return r ? r.value : null;
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
    ctx.clearRect(0, 0, this.w, this.h);
    this._handles = [];
    if (this.config.layout === 'source') { this._drawSourceLayout(ctx); return; }
    if (!this._wheelCanvas) this._buildWheel();
    ctx.drawImage(this._wheelCanvas, cx - this.size / 2, 0, this.size, this.size);   // disc, centred
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(0,0,0,0.3)'; ctx.lineWidth = 1; ctx.stroke();
    const txt = getComputedStyle(this).color || '#888';
    // Side captions out in the gutters, clear of the disc: shadows near green (left),
    // casters near red (right). Centred in the space between the rim and the edge.
    ctx.font = 'bold 12px sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    ctx.fillStyle = txt;
    const labs = this.config.labels || { left: 'Shadows', right: 'Casters' };
    if (labs.left) ctx.fillText(labs.left, (cx - R) / 2, cy);
    if (labs.right) ctx.fillText(labs.right, (cx + R + this.w) / 2, cy);
    ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI); ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fill();
    this._drawResetButton(ctx);
    const wedges = this.config.wedges || [];
    let resolved = 0;
    for (const w of wedges) {
      const c = this._val(w.chroma);
      const cone = !!w.center;          // cone: center+halfwidth edges; band: lo+hi edges
      let lo, hi, center, halfw;
      if (cone) {
        center = this._val(w.center); halfw = this._val(w.halfwidth);
        if (c == null || center == null || halfw == null) continue;
        lo = center - halfw; hi = center + halfw;
      } else {
        lo = this._val(w.lo); hi = this._val(w.hi);
        if (c == null || lo == null || hi == null) continue;   // slider not in DOM yet -> poll retries
      }
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
      const at = (ang, r) => ({ x: cx + r * Math.cos(ang), y: cy - r * Math.sin(ang) });
      if (cone) {                        // axis handle -> Target Hue; + edge -> Hue Range (halfwidth)
        this._handles.push(
          { id: w.center, type: 'angle', ref: w.ref, color: w.color, val: center, ...at(amid, rmid) },
          { id: w.halfwidth, type: 'halfwidth', ref: w.ref, center, color: w.color, val: halfw, ...at(a1, rmid) },
          { id: w.chroma, type: 'radius', ref: w.ref, color: w.color, val: c, ...at(amid, rin) });
      } else {
        this._handles.push(
          { id: w.lo, type: 'angle', ref: w.ref, color: w.color, val: lo, ...at(a0, rmid) },
          { id: w.hi, type: 'angle', ref: w.ref, color: w.color, val: hi, ...at(a1, rmid) },
          { id: w.chroma, type: 'radius', ref: w.ref, color: w.color, val: c, ...at(amid, rin) });
      }
    }
    if (this.config.sizeReach) this._drawSizeReach(ctx);   // pushes its own (custom) handles
    if (this.config.repair) this._drawRepair(ctx);
    const activeId = this._drag ? this._drag.id : this._hoverId;
    for (const h of this._handles) if (!h.custom) this._drawHandle(ctx, cx, cy, h, h.id === activeId);
    if (wedges.length && resolved === wedges.length) this._stopResolve();   // every wedge live -> stop watching
  }
  // Purple "source" layout: the SAME disc as green, carved down to just the magenta wedge --
  // real a*b* colours clipped to that slice, sitting where the green disc sat. The source-
  // highlight blob (caster: brightness = min lightness, radius = min area, ring = reach) drops
  // into the carved-away region; repair stays in the foot, exactly as on the green wheel.
  _drawSourceLayout(ctx) {
    const { cx, cy, R } = this._geom();
    if (!this._wheelCanvas) this._buildWheel();
    this._drawResetButton(ctx);
    const w = (this.config.wedges || [])[0], src = this.config.source, maxC = this.config.maxChroma || 60;
    if (!w || !src) return;
    const center = this._val(w.center), halfw = this._val(w.halfwidth), chroma = this._val(w.chroma);
    const light = this._val(src.lightness), area = this._val(src.area), reach = this._val(src.reach);
    if ([center, halfw, chroma, light, area, reach].some((v) => v == null)) return;   // slider not in DOM yet -> poll retries
    const D = Math.PI / 180;
    // Rotate the colour disc so the wedge's reference hue points screen-right: the magenta band
    // fills the right half (shadow controls), the caster blob takes the left -- mirroring green.
    const doff = (360 - (w.ref % 360)) % 360;            // screen(hue) = hue + doff;  ref -> 0deg
    const aAx = center, a0 = center - halfw, a1 = center + halfw;   // screen angles (deg): axis = Target Hue
    const rin = Math.max(0, Math.min(1, chroma / maxC)) * R;
    const at = (deg, r) => ({ x: cx + r * Math.cos(deg * D), y: cy - r * Math.sin(deg * D) });
    const wedge = (t0, t1, ri, ro) => { ctx.beginPath(); ctx.arc(cx, cy, ro, -t0 * D, -t1 * D, true); ctx.arc(cx, cy, ri, -t1 * D, -t0 * D, false); ctx.closePath(); };
    const blit = () => { ctx.save(); ctx.translate(cx, cy); ctx.rotate(-doff * D); ctx.drawImage(this._wheelCanvas, -this.size / 2, -this.size / 2, this.size, this.size); ctx.restore(); };
    // (1) the right-half colour band = the hue range you can select within, shown dimmed so the
    // current selection reads on top -- "where we're going with the hue adjustments" at a glance.
    ctx.save(); wedge(-90, 90, 0, R); ctx.clip(); blit(); ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(cx - R, cy - R, 2 * R, 2 * R); ctx.restore();
    // (2) the current selection, full colour on top
    ctx.save(); wedge(a0, a1, rin, R); ctx.clip(); blit(); ctx.restore();
    wedge(a0, a1, rin, R); ctx.lineWidth = 2.5; ctx.strokeStyle = w.color; ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI); ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fill();
    // gutter captions, mirroring green but switched: Casters (the blob, left) / Shadows (the wedge, right)
    ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = getComputedStyle(this).color || '#888';
    const labs = this.config.labels || {};
    if (labs.left) ctx.fillText(labs.left, (cx - R) / 2, cy);
    if (labs.right) ctx.fillText(labs.right, (cx + R + this.w) / 2, cy);
    // wedge handles -- standard cone types; doff is carried so drag maps screen angle -> hue
    const rmid = (rin + R) / 2;
    const hAxis = { id: w.center, type: 'angle', ref: w.ref, doff, color: w.color, val: center, ...at(aAx, R * 0.84) };   // orientation -> near the outer rim (clear of the gutter caption)
    const hHalf = { id: w.halfwidth, type: 'halfwidth', ref: w.ref, doff, center, color: w.color, val: halfw, ...at(a1, rmid) };
    const hChrm = { id: w.chroma, type: 'radius', ref: w.ref, doff, color: w.color, val: chroma, ...at(aAx, rin) };        // min chroma on the inner arc / axis, as on the green wheel
    this._handles.push(hAxis, hHalf, hChrm);
    const active = this._drag ? this._drag.id : this._hoverId;
    for (const h of [hAxis, hHalf, hChrm]) this._drawHandle(ctx, cx, cy, h, h.id === active);
    // caster blob out in the left margin, clear of the colour band, vertically aligned with the wedge
    this._drawSourceBlob(ctx, src, light, area, reach, cx - 0.78 * R, cy);
    if (this.config.repair) this._drawRepair(ctx);
    if (this.config.excessRing) this._drawExcessRing(ctx);   // ambient-centred excess-over-tone blob (right foot)
    this._stopResolve();
  }
  _drawSourceBlob(ctx, src, light, area, reach, bcx, bcy) {
    const rg = this._range(src.lightness) || { min: 0, max: 100 };
    const aMax = (this._range(src.area) || { max: 150 }).max, rMax = (this._range(src.reach) || { max: 30 }).max;
    // Brightness gauge is a FIXED-size disc (always legible); Min Area and Cast Reach are gap-rings
    // beyond it, so each is fully responsive across its whole range with no clamping floor.
    const Rg = 18, AGAP = 18, RGAP = 22;
    const areaR = Rg + this._grow(area, aMax, AGAP);                 // min-area boundary
    const ringR = areaR + this._grow(reach, rMax, RGAP);            // reach boundary
    const col = '#a8780f';   // darker gold -> more readable numbers
    ctx.save();                                          // reach ring (dashed, like the detect contour)
    ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.beginPath(); ctx.arc(bcx, bcy, ringR, 0, 2 * Math.PI); ctx.stroke();
    ctx.restore();
    ctx.beginPath(); ctx.arc(bcx, bcy, areaR, 0, 2 * Math.PI);       // min-area boundary (gold)
    ctx.lineWidth = 1.5; ctx.strokeStyle = this._rgba(col, 0.9); ctx.stroke();
    ctx.save();                                          // brightness gradient (white top -> dark bottom), fixed gauge
    ctx.beginPath(); ctx.arc(bcx, bcy, Rg, 0, 2 * Math.PI); ctx.clip();
    const g = ctx.createLinearGradient(0, bcy - Rg, 0, bcy + Rg);
    g.addColorStop(0, 'rgba(246,241,226,0.96)'); g.addColorStop(1, 'rgba(38,38,38,0.92)');
    ctx.fillStyle = g; ctx.fillRect(bcx - Rg, bcy - Rg, 2 * Rg, 2 * Rg);
    ctx.restore();
    ctx.beginPath(); ctx.arc(bcx, bcy, Rg, 0, 2 * Math.PI); ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.stroke();
    // Min Lightness as a level line across the gauge: sources above it count as casters
    const frac = (light - rg.min) / Math.max(rg.max - rg.min, 1e-6);
    const yL = bcy + Rg - frac * 2 * Rg;
    const halfW = Math.max(6, Math.sqrt(Math.max(Rg * Rg - (yL - bcy) ** 2, 0)));
    ctx.beginPath(); ctx.moveTo(bcx - halfW, yL); ctx.lineTo(bcx + halfW, yL); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.stroke();
    ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';   // small heading above the blob
    ctx.fillStyle = getComputedStyle(this).color || '#888';
    ctx.fillText('reach, lightness, & min area', bcx, Math.max(10, bcy - ringR - 30));
    const hLight = { id: src.lightness, type: 'plight', custom: true, noLabel: true, color: col, lead: col, val: light, bcy, blobR: Rg, x: bcx, y: yL };
    const hArea = { id: src.area, type: 'reach', custom: true, noLabel: true, color: col, val: area, bcx, bcy, vmax: aMax, lmax: AGAP, blobR: Rg, x: bcx + areaR, y: bcy };
    const hReach = { id: src.reach, type: 'reach', custom: true, noLabel: true, color: col, val: reach, bcx, bcy, vmax: rMax, lmax: RGAP, blobR: areaR, x: bcx - ringR, y: bcy };
    this._handles.push(hLight, hArea, hReach);
    const active = this._drag ? this._drag.id : this._hoverId;
    this._drawHandle(ctx, bcx, bcy, hLight, hLight.id === active);
    this._drawHandle(ctx, bcx, bcy, hArea, hArea.id === active);
    this._drawHandle(ctx, bcx, bcy, hReach, hReach.id === active);
    // reach / lightness / area value callouts, parked in a row above the blob, each joined to its handle
    for (const [h, cax, anchor] of [[hReach, bcx - (ringR + 4), 'right'], [hLight, bcx, 'center'], [hArea, bcx + (ringR + 4), 'left']]) {
      const cay = bcy - ringR - 14;
      ctx.strokeStyle = this._rgba(h.color, 0.6); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(h.x, h.y); ctx.lineTo(cax, cay + 5); ctx.stroke();
      const v = (Math.round(h.val * 10) / 10).toString();
      ctx.font = 'bold 11px sans-serif'; ctx.textAlign = anchor; ctx.textBaseline = 'middle';
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.strokeText(v, cax, cay);
      ctx.fillStyle = h.color; ctx.fillText(v, cax, cay);
    }
  }
  // Excess-over-ambient blob (right foot, opposite repair): a magenta disc centred on the
  // scene tone that grows magenta outward in every direction. A draggable ring = Excess
  // Threshold; what's beyond it (in any direction) is the magenta excess that gets corrected.
  _drawExcessRing(ctx) {
    const er = this.config.excessRing;
    const thr = this._val(er.threshold);
    if (thr == null) return;
    const rg = this._range(er.threshold) || { min: 0, max: 20 };
    const bcx = this.w - 78, bcy = this.size + 74, R0 = 52, RL = 44; // disc radius / threshold-ring designated max
    this._title(ctx, 'excess over ambient tone', bcx);
    const g = ctx.createRadialGradient(bcx, bcy, 0, bcx, bcy, R0);   // ambient centre -> magenta rim
    g.addColorStop(0, '#c0a878'); g.addColorStop(0.45, '#b87f8f'); g.addColorStop(1, '#b0399c');
    ctx.beginPath(); ctx.arc(bcx, bcy, R0, 0, 2 * Math.PI); ctx.fillStyle = g; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.stroke();
    const rr = Math.max(0, Math.min(R0 - 2, this._grow(thr, rg.max, RL)));
    ctx.beginPath(); ctx.arc(bcx, bcy, rr, 0, 2 * Math.PI); ctx.fillStyle = 'rgba(0,0,0,0.30)'; ctx.fill();   // spared core, dimmed
    ctx.save(); ctx.setLineDash([5, 4]); ctx.lineWidth = 2; ctx.strokeStyle = '#fff';
    ctx.beginPath(); ctx.arc(bcx, bcy, rr, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
    ctx.beginPath(); ctx.arc(bcx, bcy, 2.5, 0, 2 * Math.PI); ctx.fillStyle = '#fff'; ctx.fill();   // ambient centre
    const h = { id: er.threshold, type: 'ring', custom: true, noLabel: true, color: '#b0399c', lead: '#b0399c', val: thr, bcx, bcy, vmax: rg.max, lmax: RL, x: bcx + rr, y: bcy };
    this._handles.push(h);
    this._drawHandle(ctx, bcx, bcy, h, h.id === (this._drag ? this._drag.id : this._hoverId));
    this._callout(ctx, h, bcx + 56, this.size + 30, 'start');   // single value, above, joined by the leader
  }
  _drawSizeReach(ctx) {
    const sr = this.config.sizeReach;
    const area = this._val(sr.area), reach = this._val(sr.reach);
    if (area == null || reach == null) return;
    const bcx = this.w - 78, bcy = this.size + 74;       // blob centre, low enough to clear the title
    const aMax = (this._range(sr.area) || { max: 100 }).max, rMax = (this._range(sr.reach) || { max: 30 }).max;
    const BL = 24, GL = 30;                              // designated max radii: area disc / reach-ring gap
    const blobR = Math.max(6, this._grow(area, aMax, BL));
    const ringR = blobR + this._grow(reach, rMax, GL);
    this._title(ctx, 'reach & min size', bcx);
    ctx.save();                                          // reach ring (dashed, like the detect contour)
    ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.beginPath(); ctx.arc(bcx, bcy, ringR, 0, 2 * Math.PI); ctx.stroke();
    ctx.restore();
    ctx.beginPath(); ctx.arc(bcx, bcy, blobR, 0, 2 * Math.PI);   // red caster blob
    ctx.fillStyle = 'rgba(192,57,43,0.85)'; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.stroke();
    // reach handle on the ring's LEFT edge, size handle opposite on the blob's RIGHT edge.
    // Drawn via _drawHandle (centred on the blob) so they get the subtle tick -> hover-capsule look.
    const bh = { id: sr.area, type: 'blob', custom: true, noLabel: true, color: '#c0392b', val: area, bcx, bcy, vmax: aMax, lmax: BL,
                 x: bcx + blobR, y: bcy };
    const rh = { id: sr.reach, type: 'reach', custom: true, noLabel: true, color: '#c0392b', val: reach, bcx, bcy, vmax: rMax, lmax: GL, blobR,
                 x: bcx - ringR, y: bcy };
    this._handles.push(bh, rh);
    const active = this._drag ? this._drag.id : this._hoverId;
    this._drawHandle(ctx, bcx, bcy, bh, bh.id === active);
    this._drawHandle(ctx, bcx, bcy, rh, rh.id === active);
    // value callouts parked in the upper corners (size -> right, reach -> left), each joined
    // to its handle by a thin leader line so the numbers never collide with the ring.
    for (const [h, side] of [[bh, 1], [rh, -1]]) {
      const cax = bcx + side * 50, cay = this.size + 30;
      ctx.strokeStyle = this._rgba(h.color, 0.6); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(h.x, h.y); ctx.lineTo(cax, cay + 5); ctx.stroke();
      const v = (Math.round(h.val * 10) / 10).toString();
      ctx.font = 'bold 11px sans-serif'; ctx.textAlign = side > 0 ? 'left' : 'right'; ctx.textBaseline = 'middle';
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.strokeText(v, cax, cay);
      ctx.fillStyle = h.color; ctx.fillText(v, cax, cay);
    }
  }
  _title(ctx, s, atX) {                                // sub-panel heading, centred but kept inside the canvas
    ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const tw = ctx.measureText(s).width;
    const tx = Math.max(tw / 2 + 4, Math.min(this.w - tw / 2 - 4, atX));
    ctx.fillStyle = getComputedStyle(this).color || '#888';
    ctx.fillText(s, tx, this.size + 12);
  }
  _callout(ctx, h, cax, cay, anchor, label) {          // leader line + "label  value", parked clear of the blob
    ctx.strokeStyle = this._rgba(h.lead || '#ffffff', 0.55); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(h.x, h.y); ctx.lineTo(cax, cay + 5); ctx.stroke();
    const v = (Math.round(h.val * 100) / 100).toString();
    ctx.font = 'bold 11px sans-serif'; ctx.textAlign = anchor; ctx.textBaseline = 'middle';
    const s = label ? label + ' ' + v : v;
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.strokeText(s, cax, cay);
    ctx.fillStyle = h.color; ctx.fillText(s, cax, cay);
  }
  _drawRepair(ctx) {
    const rp = this.config.repair;
    const strength = this._val(rp.strength), spread = this._val(rp.spread), feather = this._val(rp.feather);
    if (strength == null || spread == null || feather == null) return;
    const cx = 78, cy = this.size + 74, R0 = 16;                       // base blob radius
    const sMax = (this._range(rp.spread) || { max: 5 }).max, fMax = (this._range(rp.feather) || { max: 5 }).max;
    const SL = 17, FL = 17;                                            // designated max radii for spread / feather rings
    const Rs = R0 + this._grow(spread, sMax, SL), Rf = Rs + this._grow(feather, fMax, FL);
    const G = '21,115,71', grn = '#157347';             // darker green (opposes the red size & reach)
    const col = (al) => `rgba(${G},${al})`;
    this._title(ctx, 'repair strength, spread, & feather', cx);
    ctx.save();                                          // feather: soft fading ring (the blurred edge)
    for (let k = 0; k < 3; k++) { ctx.beginPath(); ctx.arc(cx, cy, Rf - k * 1.6, 0, 2 * Math.PI); ctx.strokeStyle = col(0.5 - k * 0.14); ctx.lineWidth = 1.6; ctx.stroke(); }
    ctx.restore();
    ctx.beginPath(); ctx.arc(cx, cy, Rs, 0, 2 * Math.PI);   // spread: translucent grown region
    ctx.fillStyle = col(0.4); ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = col(0.7); ctx.stroke();
    // base shadow blob: a faint 'tube' plus a stronger fill BELOW the level line -> a strength gauge
    const yStr = cy + R0 - strength * 2 * R0;            // level: top = 1, bottom = 0
    ctx.beginPath(); ctx.arc(cx, cy, R0, 0, 2 * Math.PI); ctx.fillStyle = col(0.3); ctx.fill();
    ctx.save(); ctx.beginPath(); ctx.rect(cx - R0 - 1, yStr, 2 * R0 + 2, 2 * R0 + 2); ctx.clip();
    ctx.beginPath(); ctx.arc(cx, cy, R0, 0, 2 * Math.PI); ctx.fillStyle = col(0.95); ctx.fill(); ctx.restore();
    ctx.beginPath(); ctx.arc(cx, cy, R0, 0, 2 * Math.PI); ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.stroke();
    const halfW = Math.max(6, Math.sqrt(Math.max(R0 * R0 - (yStr - cy) ** 2, 0)));   // level line across the blob
    ctx.beginPath(); ctx.moveTo(cx - halfW, yStr); ctx.lineTo(cx + halfW, yStr); ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 1; ctx.stroke();
    const sh = { id: rp.strength, type: 'strength', custom: true, noLabel: true, color: grn, lead: grn, val: strength, cy, R0, x: cx, y: yStr };
    const spH = { id: rp.spread, type: 'spread', custom: true, noLabel: true, color: grn, lead: grn, val: spread, bcx: cx, bcy: cy, vmax: sMax, lmax: SL, R0, x: cx + Rs * Math.cos(-0.28 * Math.PI), y: cy + Rs * Math.sin(-0.28 * Math.PI) };
    const fH = { id: rp.feather, type: 'feather', custom: true, noLabel: true, color: grn, lead: grn, val: feather, bcx: cx, bcy: cy, vmax: fMax, lmax: FL, Rs, x: cx + Rf * Math.cos(0.30 * Math.PI), y: cy + Rf * Math.sin(0.30 * Math.PI) };
    this._handles.push(sh, spH, fH);
    const active = this._drag ? this._drag.id : this._hoverId;
    this._drawHandle(ctx, cx, cy, sh, sh.id === active);   // strength handle: same look as the others
    this._drawHandle(ctx, cx, cy, spH, spH.id === active);
    this._drawHandle(ctx, cx, cy, fH, fH.id === active);
    this._callout(ctx, sh, cx - 44, this.size + 30, 'end');
    this._callout(ctx, spH, cx + 50, this.size + 30, 'start');
    this._callout(ctx, fH, cx + 56, cy + 24, 'start');
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
    const radial = ['radius', 'blob', 'reach', 'spread', 'feather', 'ring'].includes(h.type);   // drags in/out
    if (!expanded) {                                     // resting: a bold tick on the wedge/circle edge
      ctx.save();
      ctx.strokeStyle = h.color; ctx.lineWidth = 5; ctx.lineCap = 'round';
      ctx.beginPath();
      if (radial) {                                      // arc segment on the circle at this radius
        const rin = Math.hypot(h.x - cx, h.y - cy);
        const amid = Math.atan2(-(h.y - cy), h.x - cx), dA = 6 / Math.max(rin, 1);
        ctx.arc(cx, cy, rin, -amid - dA, -amid + dA);
      } else if (h.type === 'strength' || h.type === 'plight') {   // horizontal tick (drags vertically)
        ctx.moveTo(h.x - 6, h.y); ctx.lineTo(h.x + 6, h.y);
      } else {                                           // segment of the radial edge through the handle
        const r = Math.hypot(h.x - cx, h.y - cy) || 1, ux = (h.x - cx) / r, uy = (h.y - cy) / r;
        ctx.moveTo(h.x - ux * 6, h.y - uy * 6); ctx.lineTo(h.x + ux * 6, h.y + uy * 6);
      }
      ctx.stroke();
      ctx.restore();
    } else {
      // Grippable capsule along the handle's drag axis: tangential for the hue edges
      // (you slide along the arc), radial for chroma (you slide in/out).
      const pa = Math.atan2(h.y - cy, h.x - cx);         // handle's angle from centre (screen space)
      const ax = (h.type === 'strength' || h.type === 'plight') ? 0 : (radial ? pa : pa + Math.PI / 2);   // strength/lightness are horizontal bars
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
    if (h.noLabel) return;                               // size/reach draw their own callouts
    // value label — nudged radially outward so it clears the line/capsule
    const dx = h.x - cx, dy = h.y - cy, len = Math.hypot(dx, dy) || 1;
    const lx = h.x + (dx / len) * 16, ly = h.y + (dy / len) * 16;
    const valTxt = (Math.round(h.val * 10) / 10).toString();
    ctx.font = 'bold 10px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.strokeText(valTxt, lx, ly);
    ctx.fillStyle = h.color; ctx.fillText(valTxt, lx, ly);
  }
}
if (!customElements.get('defringe-wheel')) customElements.define('defringe-wheel', DefringeWheel);
