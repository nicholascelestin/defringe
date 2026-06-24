// Interactive a*b* colour wheel — a controlled view: host sets `config` + `values`, wheel emits
// paraminput / paramcommit / paramreset. Host-agnostic (no DOM / Gradio / slider knowledge); the only
// file here that speaks hue / chroma / fringe / caster. Built on geometry / colour / pen.

import {
  DEG, clamp01, lerp, distance, polar, angleAt, bearing, wrapSigned, chordHalfWidth,
  verticalFraction, snapToRange, roundTo, safeSpan, growRadius, shrinkToValue, unitVectorFrom,
} from './geometry.js';
import { hexToRgba, abDiscImageData } from './colour.js';
import { Pen } from './pen.js';

const DISC_BOX = 260;            // all geometry is "drawing px" (pre-dpr); CSS px and device px differ
const DISC_MARGIN = 16;
const CAPTION_GUTTER = 64;
const FOOT_STRIP = 132;
const SOURCE_LEFT_PAD = 64;      // source layout: keep the caster blob clear of the band
const FOOT_BLOB_Y = DISC_BOX + 74;
const SIDE_BLOB_INSET = 78;
const RESET_LABEL = 'Reset';
const GRAB_RADIUS = 12;
const DISC_LIGHTNESS = 74;
const DEFAULT_MAX_CHROMA = 60;

class DefringeWheel extends HTMLElement {
  #config = {}; #pen; #layout; #wheelDisc; #values = {};
  #handles = []; #deferred = []; #drag = null; #hoverId = null; #resetHot = false; #resetBox = null;

  connectedCallback() { this.#mountWhenReady(); }

  // ── controlled-view surface (data in, events out) ──
  // Host sets both only AFTER upgrade (gradio_ui): assigning an accessor pre-upgrade leaves a
  // shadowing own property and the setter never runs.
  set config(spec) {
    const config = spec || {};
    this.#validate(config);
    this.#config = config;
    this.#layout = layoutFor(config);
    this.#wheelDisc = null;                           // maxChroma may differ -> rebuild the disc bitmap
    if (this.#pen) { this.#sizeCanvas(); this.draw(); } else this.#mountWhenReady();
  }
  get config() { return this.#config; }

  set values(snapshot) {
    this.#values = snapshot || {};
    if (this.#pen) this.draw();
  }
  get values() { return this.#values; }

  draw() {
    this.#pen.clear(this.#layout.width, this.#layout.height);
    this.#handles = []; this.#deferred = [];
    if (!this.#wheelDisc) this.#buildWheelDisc();
    if (this.#isSourceLayout()) { this.#drawSourceLayout(); return; }
    this.#drawGreenLayout();
  }

  // ── layout: green (band/cone) ──
  #drawGreenLayout() {
    const disc = this.#layout.disc;
    this.#drawDisc(disc);
    this.#drawCaptions(disc, this.config.labels || { left: 'Shadows', right: 'Casters' });
    this.#drawResetButton();
    const wedges = this.config.wedges || [];
    for (const wedge of wedges) this.#drawWedge(wedge, disc);
    if (this.config.sizeReach) this.#drawSizeReach();
    if (this.config.repair) this.#drawRepair();
    for (const handle of this.#deferred) { this.#drawMark(disc.center, handle); this.#labelHandle(disc.center, handle); }
  }

  #drawDisc({ center, radius }) {
    this.#pen.blit(this.#wheelDisc, center.x - DISC_BOX / 2, 0, DISC_BOX);
    this.#pen.strokeCircle(center, radius, 'rgba(0,0,0,0.3)', 1);
    this.#pen.dot(center, 2.5, 'rgba(0,0,0,0.45)');
  }

  #drawWedge(wedge, disc) {
    const lo = this.#value(wedge.lo), hi = this.#value(wedge.hi), chroma = this.#value(wedge.chroma);
    if (lo == null || hi == null || chroma == null) return;
    const { center, radius } = disc;
    const loAngle = (wedge.ref + lo) * DEG, hiAngle = (wedge.ref + hi) * DEG;
    const innerRadius = clamp01(chroma / this.#maxChroma()) * radius;
    this.#pen.sector(center, innerRadius, radius, loAngle, hiAngle,
      { fill: hexToRgba(wedge.color, 0.12), stroke: wedge.color, lineWidth: 2.5 });
    const midRadius = (innerRadius + radius) / 2, midAngle = (loAngle + hiAngle) / 2;
    const at = (angle, atRadius) => polar(center, angle, atRadius);
    const h = this.#wheelHandles(wedge.color, wedge.ref);
    this.#deferWheelHandles([
      h.edge(wedge.lo, lo, at(loAngle, midRadius)),
      h.edge(wedge.hi, hi, at(hiAngle, midRadius)),
      h.chroma(wedge.chroma, chroma, at(midAngle, innerRadius)),
    ]);
  }

  // ── layout: purple "source" ──
  #drawSourceLayout() {
    const sourceParams = this.#readSourceParams();
    if (!sourceParams) return;
    const disc = this.#layout.disc;
    this.#drawResetButton();
    this.#paintCarvedDisc(disc, sourceParams);
    this.#drawCaptions(disc, this.config.labels || {});
    this.#drawSourceWedgeHandles(disc, sourceParams);
    this.#drawSourceBlob(sourceParams, { x: disc.center.x - 0.78 * disc.radius, y: disc.center.y });
    if (this.config.repair) this.#drawRepair();
    if (this.config.excessRing) this.#drawExcessRing();
  }

  #readSourceParams() {
    const wedge = (this.config.wedges || [])[0], source = this.config.source;
    if (!wedge || !source) return null;
    const center = this.#value(wedge.center), halfWidth = this.#value(wedge.halfwidth), chroma = this.#value(wedge.chroma);
    const lightness = this.#value(source.lightness), area = this.#value(source.area), reach = this.#value(source.reach);
    if ([center, halfWidth, chroma, lightness, area, reach].some((v) => v == null)) return null;
    return {
      wedge, source, center, halfWidth, chroma, lightness, area, reach,
      displayOffset: (360 - (wedge.ref % 360)) % 360,      // screen(hue) = hue + displayOffset; ref -> 0deg
      axisDeg: center, loDeg: center - halfWidth, hiDeg: center + halfWidth,
      innerRadius: clamp01(chroma / this.#maxChroma()) * this.#layout.disc.radius,
    };
  }

  // ref hue rotated to screen-right (mirrors green): band fills the right half, blob takes the left.
  #paintCarvedDisc({ center, radius }, sourceParams) {
    const blitRotated = () => this.#pen.blitRotated(this.#wheelDisc, center, -sourceParams.displayOffset * DEG, DISC_BOX);
    this.#pen.clippedToSector(center, 0, radius, -90 * DEG, 90 * DEG, () => {
      blitRotated();
      this.#pen.fillBox(center, radius, 'rgba(0,0,0,0.5)');
    });
    this.#pen.clippedToSector(center, sourceParams.innerRadius, radius, sourceParams.loDeg * DEG, sourceParams.hiDeg * DEG, blitRotated);
    this.#pen.sector(center, sourceParams.innerRadius, radius, sourceParams.loDeg * DEG, sourceParams.hiDeg * DEG,
      { stroke: sourceParams.wedge.color, lineWidth: 2.5 });
    this.#pen.dot(center, 2.5, 'rgba(0,0,0,0.45)');
  }

  #drawSourceWedgeHandles(disc, sourceParams) {
    const { wedge, center, halfWidth, chroma, axisDeg, hiDeg, innerRadius, displayOffset } = sourceParams;
    const at = (deg, radius) => polar(disc.center, deg * DEG, radius);
    const midRadius = (innerRadius + disc.radius) / 2;
    const h = this.#wheelHandles(wedge.color, wedge.ref, displayOffset);   // offset carried so drag maps screen angle -> hue
    const handles = [
      h.edge(wedge.center, center, at(axisDeg, disc.radius * 0.84)),
      h.span(wedge.halfwidth, halfWidth, center, at(hiDeg, midRadius)),
      h.chroma(wedge.chroma, chroma, at(axisDeg, innerRadius)),
    ];
    for (const handle of handles) { this.#handles.push(handle); this.#drawMark(disc.center, handle); this.#labelHandle(disc.center, handle); }
  }

  // ── foot gauges ──
  #drawSizeReach() {
    const sizeReach = this.config.sizeReach;
    const area = this.#value(sizeReach.area), reach = this.#value(sizeReach.reach);
    if (area == null || reach == null) return;
    const blob = this.#layout.footRight;
    const areaMax = this.#rangeMax(sizeReach.area, 100), reachMax = this.#rangeMax(sizeReach.reach, 30);
    const areaDiscMax = 24, reachGapMax = 30;
    const blobRadius = Math.max(6, growRadius(area, areaMax, areaDiscMax));
    const ringRadius = blobRadius + growRadius(reach, reachMax, reachGapMax);
    this.#title('reach & min size', blob.x);
    this.#pen.dashedRing(blob, ringRadius);
    this.#pen.dot(blob, blobRadius, 'rgba(192,57,43,0.85)', 'rgba(255,255,255,0.5)', 1);
    const h = this.#blobHandles(blob, '#c0392b');
    const size = h.gap({ id: sizeReach.area, value: area, inner: 0, valueMax: areaMax, radiusMax: areaDiscMax, at: polar(blob, 0, blobRadius) });
    const reach2 = h.gap({ id: sizeReach.reach, value: reach, inner: blobRadius, valueMax: reachMax, radiusMax: reachGapMax, at: polar(blob, Math.PI, ringRadius) });
    this.#callout(size, blob.x + 50, DISC_BOX + 30, 'left');
    this.#callout(reach2, blob.x - 50, DISC_BOX + 30, 'right');
  }

  #drawSourceBlob(sourceParams, blob) {
    const source = sourceParams.source;
    const lightRange = this.#range(source.lightness) || { min: 0, max: 100 };
    const areaMax = this.#rangeMax(source.area, 150), reachMax = this.#rangeMax(source.reach, 30);
    const gaugeRadius = 18, areaGapMax = 18, reachGapMax = 22;
    const areaRadius = gaugeRadius + growRadius(sourceParams.area, areaMax, areaGapMax);
    const ringRadius = areaRadius + growRadius(sourceParams.reach, reachMax, reachGapMax);
    const gold = '#a8780f';
    this.#pen.dashedRing(blob, ringRadius);
    this.#pen.strokeCircle(blob, areaRadius, hexToRgba(gold, 0.9), 1.5);
    this.#brightnessGauge(blob, gaugeRadius);
    const fraction = (sourceParams.lightness - lightRange.min) / safeSpan(lightRange.max - lightRange.min);
    const levelY = this.#levelLine(blob, gaugeRadius, fraction, '#fff', 1.4);   // above this lightness, a source counts as a caster
    this.#title('reach, lightness, & min area', blob.x, blob.y - ringRadius - 30);
    const h = this.#blobHandles(blob, gold);
    const light = h.rangeLevel({ id: source.lightness, value: sourceParams.lightness, radius: gaugeRadius, at: { x: blob.x, y: levelY } });
    const area = h.gap({ id: source.area, value: sourceParams.area, inner: gaugeRadius, valueMax: areaMax, radiusMax: areaGapMax, at: polar(blob, 0, areaRadius) });
    const reach = h.gap({ id: source.reach, value: sourceParams.reach, inner: areaRadius, valueMax: reachMax, radiusMax: reachGapMax, at: polar(blob, Math.PI, ringRadius) });
    const calloutY = blob.y - ringRadius - 14;
    this.#callout(reach, blob.x - (ringRadius + 4), calloutY, 'right');
    this.#callout(light, blob.x, calloutY, 'center');
    this.#callout(area, blob.x + (ringRadius + 4), calloutY, 'left');
  }

  // the dashed ring is the threshold: tone beyond it (any direction from ambient) is the corrected excess.
  #drawExcessRing() {
    const excessRing = this.config.excessRing;
    const threshold = this.#value(excessRing.threshold);
    if (threshold == null) return;
    const range = this.#range(excessRing.threshold) || { min: 0, max: 20 };
    const blob = this.#layout.footRight, discRadius = 52, thresholdMax = 44;
    this.#title('excess over ambient tone', blob.x);
    this.#pen.radialGradientDisc(blob, discRadius, [[0, '#c0a878'], [0.45, '#b87f8f'], [1, '#b0399c']], 'rgba(255,255,255,0.5)', 1);
    const thresholdRadius = Math.max(0, Math.min(discRadius - 2, growRadius(threshold, range.max, thresholdMax)));
    this.#pen.dot(blob, thresholdRadius, 'rgba(0,0,0,0.30)');
    this.#pen.dashedRing(blob, thresholdRadius, [5, 4], 2, '#fff');
    this.#pen.dot(blob, 2.5, '#fff');
    const handle = this.#blobHandles(blob, '#b0399c').gap({ id: excessRing.threshold, value: threshold, inner: 0, valueMax: range.max, radiusMax: thresholdMax, at: polar(blob, 0, thresholdRadius) });
    this.#callout(handle, blob.x + 56, DISC_BOX + 30, 'start', { decimals: 2 });
  }

  #drawRepair() {
    const repair = this.config.repair;
    const strength = this.#value(repair.strength), spread = this.#value(repair.spread), feather = this.#value(repair.feather);
    if (strength == null || spread == null || feather == null) return;
    const blob = this.#layout.footLeft, baseRadius = 16;
    const spreadMax = this.#rangeMax(repair.spread, 5), featherMax = this.#rangeMax(repair.feather, 5);
    const spreadGapMax = 17, featherGapMax = 17;
    const spreadRadius = baseRadius + growRadius(spread, spreadMax, spreadGapMax);
    const featherRadius = spreadRadius + growRadius(feather, featherMax, featherGapMax);
    const green = '#157347', channels = '21,115,71';
    this.#title('repair spread, strength, & feather', blob.x);
    this.#featherRings(blob, featherRadius, channels);
    this.#pen.dot(blob, spreadRadius, `rgba(${channels},0.4)`, `rgba(${channels},0.7)`, 1);
    const levelY = this.#fillGauge(blob, baseRadius, strength, channels);
    const h = this.#blobHandles(blob, green);
    const strengthHandle = h.level({ id: repair.strength, value: strength, radius: baseRadius, at: { x: blob.x, y: levelY } });
    const spreadHandle = h.gap({ id: repair.spread, value: spread, inner: baseRadius, valueMax: spreadMax, radiusMax: spreadGapMax, at: polar(blob, Math.PI, spreadRadius) });
    const featherHandle = h.gap({ id: repair.feather, value: feather, inner: spreadRadius, valueMax: featherMax, radiusMax: featherGapMax, at: polar(blob, 0, featherRadius) });
    this.#callout(spreadHandle, blob.x - 50, DISC_BOX + 30, 'right', { decimals: 2 });
    this.#callout(strengthHandle, blob.x, DISC_BOX + 30, 'center', { decimals: 2 });
    this.#callout(featherHandle, blob.x + 50, DISC_BOX + 30, 'left', { decimals: 2 });
  }

  // ── gauges (generic pen verbs, coloured here where the meaning lives) ──
  #brightnessGauge(blob, radius) {
    this.#pen.verticalGradientDisc(blob, radius, 'rgba(246,241,226,0.96)', 'rgba(38,38,38,0.92)');
    this.#pen.strokeCircle(blob, radius, 'rgba(255,255,255,0.6)', 1);
  }
  #fillGauge(blob, radius, fraction, channels) {
    const levelY = blob.y + radius - fraction * 2 * radius;
    this.#pen.dot(blob, radius, `rgba(${channels},0.3)`);
    this.#pen.fillCircleBelow(blob, radius, levelY, `rgba(${channels},0.95)`);
    this.#pen.strokeCircle(blob, radius, 'rgba(255,255,255,0.4)', 1);
    return this.#levelLine(blob, radius, fraction, 'rgba(255,255,255,0.55)', 1);
  }
  #featherRings(blob, outerRadius, channels) {
    for (let ring = 0; ring < 3; ring++)
      this.#pen.strokeCircle(blob, outerRadius - ring * 1.6, `rgba(${channels},${0.5 - ring * 0.14})`, 1.6);
  }
  #levelLine(blob, radius, fraction, color, lineWidth) {
    const y = blob.y + radius - fraction * 2 * radius;
    const half = Math.max(6, chordHalfWidth(radius, y - blob.y));
    this.#pen.line(blob.x - half, y, blob.x + half, y, color, lineWidth);
    return y;
  }

  // ── chrome ──
  #drawCaptions({ center, radius }, labels) {
    const color = this.#textColor();
    if (labels.left) this.#pen.text(labels.left, (center.x - radius) / 2, center.y, { font: 'bold 12px sans-serif', align: 'center', fill: color });
    if (labels.right) this.#pen.text(labels.right, (center.x + radius + this.#layout.width) / 2, center.y, { font: 'bold 12px sans-serif', align: 'center', fill: color });
  }
  #title(text, atX, atY = DISC_BOX + 12) {
    const halfWidth = this.#pen.measure(text, 'bold 11px sans-serif') / 2;
    const x = Math.max(halfWidth + 4, Math.min(this.#layout.width - halfWidth - 4, atX));
    this.#pen.text(text, x, Math.max(10, atY), { font: 'bold 11px sans-serif', align: 'center', fill: this.#textColor() });
  }
  #callout(handle, textX, textY, anchor, { label, decimals = 1 } = {}) {
    this.#pen.line(handle.x, handle.y, textX, textY + 5, hexToRgba(handle.color, 0.55), 1);
    const value = roundTo(handle.value, decimals).toString();
    this.#pen.haloText(label ? label + ' ' + value : value, textX, textY, { font: 'bold 11px sans-serif', align: anchor, fill: handle.color });
  }
  #drawResetButton() {
    const iconRadius = 6, padX = 8, gap = 5, height = 22;
    const labelWidth = this.#pen.measure(RESET_LABEL, 'bold 11px sans-serif');
    const width = padX + iconRadius * 2 + gap + labelWidth + padX;
    const x1 = this.#layout.width - 6, x0 = x1 - width, y0 = 6, y1 = y0 + height;
    this.#resetBox = { x0, y0, x1, y1 };
    this.#pen.pill(x0, y0, width, height,
      this.#resetHot ? 'rgba(20,20,20,0.85)' : 'rgba(20,20,20,0.6)',
      this.#resetHot ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.3)');
    const iconY = (y0 + y1) / 2;
    this.#pen.refreshIcon(x0 + padX + iconRadius, iconY, iconRadius, '#f5f5f5');
    this.#pen.text(RESET_LABEL, x0 + padX + iconRadius * 2 + gap, iconY + 0.5, { font: 'bold 11px sans-serif', align: 'left', fill: '#f5f5f5' });
  }

  // ── handle placement ──
  #placeHandle(center, handle) { this.#handles.push(handle); this.#drawMark(center, handle); return handle; }
  #deferWheelHandles(handles) { for (const handle of handles) { this.#handles.push(handle); this.#deferred.push(handle); } }
  #drawMark(center, handle) { drawHandle(this.#pen, center, handle, handle.id === this.#activeId()); }

  // capture each gauge's shared blob + colour; the returned verbs build, register, AND draw (foot
  // gauges paint inline). Each call names only what varies.
  #blobHandles(blob, color) {
    const place = (handle) => this.#placeHandle(blob, handle);
    return {
      gap: (fields) => place(gapHandle({ ...fields, color, anchor: blob })),
      level: (fields) => place(levelHandle({ ...fields, color, center: blob })),
      rangeLevel: (fields) => place(rangeLevelHandle({ ...fields, color, center: blob })),
    };
  }
  // capture each wedge's shared colour + ref hue (+ source rotation); the verbs only BUILD — the
  // caller defers (green) or places (source).
  #wheelHandles(color, ref, displayOffset) {
    const scale = this.#maxChroma();
    return {
      edge: (id, value, at) => angleHandle({ id, value, color, ref, displayOffset, at }),
      span: (id, value, axis, at) => angleSpanHandle({ id, value, color, ref, axis, displayOffset, at }),
      chroma: (id, value, at) => radialHandle({ id, value, color, scale, at }),
    };
  }
  #activeId() { return this.#drag ? this.#drag.id : this.#hoverId; }
  #labelHandle(center, handle) {                            // nudge the value text radially out, clear of the grip
    const out = unitVectorFrom(center, handle);
    this.#pen.haloText(roundTo(handle.value, 1).toString(), handle.x + out.x * 16, handle.y + out.y * 16,
      { font: 'bold 10px sans-serif', align: 'center', fill: handle.color });
  }

  // ── setup: mount once we're both connected AND configured (config may arrive either side of connect) ──
  #mountWhenReady() {
    if (this.#pen || !this.isConnected || !this.#layout) return;
    this.#mountCanvas();
    this.#bindPointer();
    this.draw();
  }
  #mountCanvas() {
    const canvas = document.createElement('canvas');
    canvas.style.maxWidth = '100%'; canvas.style.height = 'auto';
    canvas.style.display = 'block'; canvas.style.margin = '0 auto';
    canvas.style.touchAction = 'none';
    this.style.display = 'block';
    this.appendChild(canvas);
    this.#pen = new Pen(canvas.getContext('2d'), window.devicePixelRatio || 1);
    this.#sizeCanvas();
  }
  #sizeCanvas() {                                    // setTransform (not scale) so a reconfigure can't compound the dpr
    const { canvas, ctx, dpr } = this.#pen;
    canvas.width = this.#layout.width * dpr; canvas.height = this.#layout.height * dpr;
    canvas.style.width = this.#layout.width + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  #bindPointer() {
    const canvas = this.#pen.canvas;
    canvas.addEventListener('pointerdown', (e) => this.#onPointerDown(e));
    canvas.addEventListener('pointermove', (e) => this.#onPointerMove(e));
    canvas.addEventListener('pointerup', () => this.#onPointerUp());
    canvas.addEventListener('pointercancel', () => this.#onPointerUp());
    canvas.addEventListener('pointerleave', () => this.#onPointerLeave());
  }

  // ── pointer interaction ──
  #onPointerDown(e) {
    const point = this.#pointerXY(e);
    if (this.#overReset(point.x, point.y)) { this.#reset(); e.preventDefault(); return; }
    const handle = nearestHandle(this.#handles, point, GRAB_RADIUS);
    if (!handle) return;
    this.#drag = handle;
    this.#pen.canvas.setPointerCapture(e.pointerId);
    this.#pen.cursor('grabbing');
    e.preventDefault();
  }
  #onPointerMove(e) {
    const point = this.#pointerXY(e);
    if (this.#drag) { this.#dragTo(point); e.preventDefault(); return; }
    this.#hoverAt(point);
  }
  #dragTo(point) {
    const handle = this.#drag;
    const range = this.#range(handle.id);
    if (!range) return;
    const env = { center: this.#layout.disc.center, radius: this.#layout.disc.radius, range };
    const value = snapToRange(projectPointer(handle, point, env), range);
    this.#echo(handle.id, value);                          // local-echo so the drag is smooth without a host round-trip
    this.draw();
    this.#emit('paraminput', { id: handle.id, value });
  }
  #hoverAt(point) {
    const onReset = this.#overReset(point.x, point.y);
    const handle = onReset ? null : nearestHandle(this.#handles, point, GRAB_RADIUS);
    this.#pen.cursor(onReset ? 'pointer' : (handle ? 'grab' : 'default'));
    const id = handle ? handle.id : null;
    if (id !== this.#hoverId || onReset !== this.#resetHot) {
      this.#hoverId = id; this.#resetHot = onReset; this.draw();
    }
  }
  #onPointerUp() {
    if (!this.#drag) return;
    const id = this.#drag.id;
    this.#drag = null;
    this.#pen.cursor('grab');
    this.#emit('paramcommit', { id, value: this.#value(id) });   // finalize -> host re-runs the pipeline
  }
  #onPointerLeave() {
    if (this.#drag) return;
    this.#pen.cursor('default');
    let redraw = this.#resetHot; this.#resetHot = false;
    if (this.#hoverId !== null) { this.#hoverId = null; redraw = true; }
    if (redraw) this.draw();
  }
  #pointerXY(e) {
    const rect = this.#pen.canvas.getBoundingClientRect();  // map client px -> drawing px (canvas may be CSS-scaled)
    return { x: (e.clientX - rect.left) * (this.#layout.width / rect.width), y: (e.clientY - rect.top) * (this.#layout.height / rect.height) };
  }
  #overReset(x, y) {
    const box = this.#resetBox;
    return box && x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1;
  }
  #reset() { this.#emit('paramreset', {}); }

  // ── view state ──
  #isSourceLayout() { return this.config.layout === 'source'; }
  #maxChroma() { return this.config.maxChroma || DEFAULT_MAX_CHROMA; }
  #textColor() { return getComputedStyle(this).color || '#888'; }
  #value(id) { const entry = this.#values[id]; return entry ? entry.value : null; }
  #range(id) { const entry = this.#values[id]; return entry ? { min: entry.min, max: entry.max, step: entry.step } : null; }
  #rangeMax(id, fallback) { return (this.#range(id) || { max: fallback }).max; }
  #echo(id, value) { this.#values = { ...this.#values, [id]: { ...this.#values[id], value } }; }
  #emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail, bubbles: true })); }
  #buildWheelDisc() {
    const dpr = this.#pen.dpr;
    const diameter = Math.round(DISC_BOX * dpr), radius = (DISC_BOX / 2 - DISC_MARGIN) * dpr;
    this.#wheelDisc = this.#pen.bitmap(abDiscImageData(diameter, radius, this.#maxChroma(), DISC_LIGHTNESS));
  }

  // ── the config contract: the wheel's own definition of a well-formed config ──
  // Every lower-case leaf is an opaque param key; values[key] carries its { value, min, max, step }.
  static #GAUGE_BLOCKS = {
    sizeReach: ['area', 'reach'],
    source: ['lightness', 'area', 'reach'],
    excessRing: ['threshold'],
    repair: ['strength', 'spread', 'feather'],
  };
  #validate(config) {
    if (!Array.isArray(config.wedges) || config.wedges.length === 0)
      throw new Error('<defringe-wheel>: config.wedges must be a non-empty array');
    for (const wedge of config.wedges) {
      const band = wedge.lo != null && wedge.hi != null;
      const cone = wedge.center != null && wedge.halfwidth != null;
      if (wedge.ref == null || !wedge.color || wedge.chroma == null || !(band || cone))
        throw new Error('<defringe-wheel>: each wedge needs { ref, color, chroma } and band { lo, hi } or cone { center, halfwidth }');
    }
    for (const [block, keys] of Object.entries(DefringeWheel.#GAUGE_BLOCKS))
      if (config[block] && !keys.every((key) => config[block][key] != null))
        throw new Error(`<defringe-wheel>: config.${block} needs { ${keys.join(', ')} }`);
  }
}
if (!customElements.get('defringe-wheel')) customElements.define('defringe-wheel', DefringeWheel);

// ── the handle model: a draggable point that maps a pointer to a value (domain-free) ──
// Kinds are GEOMETRIC, never colour — the wheel alone decides radial=chroma, angle=hue.
const HANDLE_KINDS = {
  radial:     { orient: 'radial',     toValue: (h, p, env) => clamp01(distance(p, env.center) / env.radius) * h.scale },
  angle:      { orient: 'edge',       toValue: (h, p, env) => wrapSigned(angleAt(p, env.center) - h.ref - (h.displayOffset || 0)) },
  angleSpan:  { orient: 'edge',       toValue: (h, p, env) => Math.abs(wrapSigned(angleAt(p, env.center) - (h.ref + h.axis + (h.displayOffset || 0)))) },
  gap:        { orient: 'radial',     toValue: (h, p) => shrinkToValue(Math.max(0, distance(p, h.anchor) - h.inner), h.valueMax, h.radiusMax) },
  level:      { orient: 'horizontal', toValue: (h, p) => verticalFraction(p.y, h.center.y, h.radius) },
  rangeLevel: { orient: 'horizontal', toValue: (h, p, env) => lerp(env.range.min, env.range.max, verticalFraction(p.y, h.center.y, h.radius)) },
};
function projectPointer(handle, point, env) { return HANDLE_KINDS[handle.kind].toValue(handle, point, env); }
function orientOf(handle) { return HANDLE_KINDS[handle.kind].orient; }
function nearestHandle(handles, point, grabRadius) {
  let best = null, bestDist = grabRadius * grabRadius;
  for (const handle of handles) {
    const d = (handle.x - point.x) ** 2 + (handle.y - point.y) ** 2;
    if (d <= bestDist) { bestDist = d; best = handle; }
  }
  return best;
}
function drawHandle(pen, center, handle, active) {
  const orient = orientOf(handle);
  if (active) {
    const fromCenter = Math.atan2(handle.y - center.y, handle.x - center.x);
    const angle = orient === 'horizontal' ? 0 : (orient === 'radial' ? fromCenter : fromCenter + Math.PI / 2);
    pen.gripCapsule(handle, angle, handle.color);
  } else if (orient === 'radial') {
    const radius = distance(handle, center);
    pen.arcSegment(center, radius, bearing(handle, center), 6 / Math.max(radius, 1), handle.color, 5);
  } else if (orient === 'horizontal') {                     // horizontal tick, but drags vertically
    pen.segment(handle.x - 6, handle.y, handle.x + 6, handle.y, handle.color, 5);
  } else {
    const out = unitVectorFrom(center, handle);
    pen.segment(handle.x - out.x * 6, handle.y - out.y * 6, handle.x + out.x * 6, handle.y + out.y * 6, handle.color, 5);
  }
}

// One factory per kind; `at` is the screen anchor {x, y}.
function radialHandle({ id, value, color, scale, at }) {
  return { kind: 'radial', id, value, color, scale, x: at.x, y: at.y };
}
function angleHandle({ id, value, color, ref, displayOffset, at }) {
  return { kind: 'angle', id, value, color, ref, displayOffset, x: at.x, y: at.y };
}
function angleSpanHandle({ id, value, color, ref, axis, displayOffset, at }) {
  return { kind: 'angleSpan', id, value, color, ref, axis, displayOffset, x: at.x, y: at.y };
}
function gapHandle({ id, value, color, anchor, inner, valueMax, radiusMax, at }) {
  return { kind: 'gap', id, value, color, anchor, inner, valueMax, radiusMax, x: at.x, y: at.y };
}
function levelHandle({ id, value, color, center, radius, at }) {
  return { kind: 'level', id, value, color, center, radius, x: at.x, y: at.y };
}
function rangeLevelHandle({ id, value, color, center, radius, at }) {
  return { kind: 'rangeLevel', id, value, color, center, radius, x: at.x, y: at.y };
}

// ── layout: the single home for "how big is everything", as named regions ──
function layoutFor(config) {
  const foot = (config.sizeReach || config.repair) ? FOOT_STRIP : 0;
  const padLeft = (config.layout === 'source') ? SOURCE_LEFT_PAD : 0;
  const width = DISC_BOX + 2 * CAPTION_GUTTER + padLeft;
  const height = DISC_BOX + foot;
  return {
    width, height,
    disc: { center: { x: DISC_BOX / 2 + CAPTION_GUTTER + padLeft, y: DISC_BOX / 2 }, radius: DISC_BOX / 2 - DISC_MARGIN },
    footLeft: { x: SIDE_BLOB_INSET, y: FOOT_BLOB_Y },
    footRight: { x: width - SIDE_BLOB_INSET, y: FOOT_BLOB_Y },
  };
}