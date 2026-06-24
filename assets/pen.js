// A vocabulary of drawing verbs over a 2D canvas context. Every argument is a point {x,y}, a number,
// a colour string, gradient stops, or ImageData — no colour science, no defringe, no handle concept.

import { TAU } from './geometry.js';

export class Pen {
  constructor(ctx, dpr) { this.ctx = ctx; this.dpr = dpr; this.canvas = ctx.canvas; }

  clear(width, height) { this.ctx.clearRect(0, 0, width, height); }
  cursor(name) { this.canvas.style.cursor = name; }
  measure(text, font) { this.ctx.font = font; return this.ctx.measureText(text).width; }

  // ── circles & lines ──
  dot(center, radius, fill, stroke, strokeWidth) {
    const ctx = this.ctx;
    ctx.beginPath(); ctx.arc(center.x, center.y, radius, 0, TAU);
    ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.lineWidth = strokeWidth; ctx.strokeStyle = stroke; ctx.stroke(); }
  }
  strokeCircle(center, radius, stroke, lineWidth) {
    const ctx = this.ctx;
    ctx.beginPath(); ctx.arc(center.x, center.y, radius, 0, TAU);
    ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke();
  }
  dashedRing(center, radius, dash = [4, 3], lineWidth = 1.5, stroke = 'rgba(255,255,255,0.85)') {
    const ctx = this.ctx;
    ctx.save(); ctx.setLineDash(dash); ctx.lineWidth = lineWidth; ctx.strokeStyle = stroke;
    ctx.beginPath(); ctx.arc(center.x, center.y, radius, 0, TAU); ctx.stroke();
    ctx.restore();
  }
  fillBox(center, halfSize, fill) {
    this.ctx.fillStyle = fill;
    this.ctx.fillRect(center.x - halfSize, center.y - halfSize, 2 * halfSize, 2 * halfSize);
  }
  line(x0, y0, x1, y1, stroke, lineWidth) {
    const ctx = this.ctx;
    ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
  segment(x0, y0, x1, y1, stroke, lineWidth) {             // round-capped stroke
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.restore();
  }
  arcSegment(center, radius, midAngle, halfSpan, stroke, lineWidth) {   // round-capped arc; angle math-space (y up)
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.arc(center.x, center.y, radius, -midAngle - halfSpan, -midAngle + halfSpan); ctx.stroke();
    ctx.restore();
  }

  // ── sectors (annular wedges) ──
  sector(center, innerR, outerR, loAngle, hiAngle, { fill, stroke, lineWidth } = {}) {
    this.#sectorPath(center, innerR, outerR, loAngle, hiAngle);
    if (fill) { this.ctx.fillStyle = fill; this.ctx.fill(); }
    if (stroke) { this.ctx.lineWidth = lineWidth; this.ctx.strokeStyle = stroke; this.ctx.stroke(); }
  }
  clippedToSector(center, innerR, outerR, loAngle, hiAngle, draw) {
    this.ctx.save();
    this.#sectorPath(center, innerR, outerR, loAngle, hiAngle); this.ctx.clip();
    draw();
    this.ctx.restore();
  }
  #sectorPath(center, innerR, outerR, loAngle, hiAngle) {   // canvas y is down, so angles negate
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(center.x, center.y, outerR, -loAngle, -hiAngle, true);
    ctx.arc(center.x, center.y, innerR, -hiAngle, -loAngle, false);
    ctx.closePath();
  }

  // ── grip capsule (the active-drag affordance) ──
  gripCapsule(point, angleRad, fill) {
    const ctx = this.ctx, halfLength = 8, halfWidth = 4.5;
    ctx.save();
    ctx.translate(point.x, point.y); ctx.rotate(angleRad);
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(-halfLength, -halfWidth, 2 * halfLength, 2 * halfWidth, halfWidth);
    else { ctx.arc(-halfLength + halfWidth, 0, halfWidth, Math.PI / 2, -Math.PI / 2); ctx.arc(halfLength - halfWidth, 0, halfWidth, -Math.PI / 2, Math.PI / 2); ctx.closePath(); }
    ctx.shadowColor = 'rgba(0,0,0,0.4)'; ctx.shadowBlur = 3; ctx.shadowOffsetY = 1;
    ctx.fillStyle = fill; ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.lineWidth = 1.5; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1;
    for (const tick of [-2.5, 0, 2.5]) { ctx.beginPath(); ctx.moveTo(tick, -2); ctx.lineTo(tick, 2); ctx.stroke(); }
    ctx.restore();
  }

  // ── bitmaps ──
  blit(image, x, y, size) { this.ctx.drawImage(image, x, y, size, size); }
  blitRotated(image, center, angleRad, size) {
    const ctx = this.ctx;
    ctx.save(); ctx.translate(center.x, center.y); ctx.rotate(angleRad);
    ctx.drawImage(image, -size / 2, -size / 2, size, size); ctx.restore();
  }
  bitmap(imageData) {                                       // pixel data -> an offscreen canvas, ready to blit
    const off = document.createElement('canvas');
    off.width = imageData.width; off.height = imageData.height;
    off.getContext('2d').putImageData(imageData, 0, 0);
    return off;
  }

  // ── text ──
  text(content, x, y, { font, align, fill }) {
    const ctx = this.ctx;
    ctx.font = font; ctx.textAlign = align; ctx.textBaseline = 'middle';
    ctx.fillStyle = fill; ctx.fillText(content, x, y);
  }
  haloText(content, x, y, { font, align, fill }) {
    const ctx = this.ctx;
    ctx.font = font; ctx.textAlign = align; ctx.textBaseline = 'middle';
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.strokeText(content, x, y);
    ctx.fillStyle = fill; ctx.fillText(content, x, y);
  }

  // ── chrome & gradients ──
  pill(x, y, width, height, fill, stroke) {
    const ctx = this.ctx;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, width, height, height / 2);
    else ctx.rect(x, y, width, height);
    ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = stroke; ctx.stroke();
  }
  refreshIcon(x, y, radius, color) {                        // a 270° arc with an arrowhead at its tip
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.6; ctx.lineCap = 'round';
    const start = -0.25 * Math.PI, end = 1.25 * Math.PI;
    ctx.beginPath(); ctx.arc(x, y, radius, start, end); ctx.stroke();
    arrowHead(ctx, { x: x + radius * Math.cos(end), y: y + radius * Math.sin(end) }, end + Math.PI / 2, 4, 3);
    ctx.restore();
  }
  verticalGradientDisc(center, radius, topColor, bottomColor) {
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath(); ctx.arc(center.x, center.y, radius, 0, TAU); ctx.clip();
    const gradient = ctx.createLinearGradient(0, center.y - radius, 0, center.y + radius);
    gradient.addColorStop(0, topColor); gradient.addColorStop(1, bottomColor);
    ctx.fillStyle = gradient; ctx.fillRect(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
    ctx.restore();
  }
  fillCircleBelow(center, radius, y, fill) {                // the part of a circle below a horizontal line
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath(); ctx.rect(center.x - radius - 1, y, 2 * radius + 2, 2 * radius + 2); ctx.clip();
    ctx.beginPath(); ctx.arc(center.x, center.y, radius, 0, TAU); ctx.fillStyle = fill; ctx.fill();
    ctx.restore();
  }
  radialGradientDisc(center, radius, stops, stroke, strokeWidth) {
    const ctx = this.ctx;
    const gradient = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, radius);
    for (const [offset, color] of stops) gradient.addColorStop(offset, color);
    this.dot(center, radius, gradient, stroke, strokeWidth);
  }
}

function arrowHead(ctx, tip, alongAngle, length, halfWidth) {
  const normalX = Math.cos(alongAngle + Math.PI / 2), normalY = Math.sin(alongAngle + Math.PI / 2);
  ctx.beginPath();
  ctx.moveTo(tip.x + length * Math.cos(alongAngle), tip.y + length * Math.sin(alongAngle));
  ctx.lineTo(tip.x + halfWidth * normalX, tip.y + halfWidth * normalY);
  ctx.lineTo(tip.x - halfWidth * normalX, tip.y - halfWidth * normalY);
  ctx.closePath(); ctx.fill();
}
