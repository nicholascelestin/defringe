// Pure scalar/vector math and value↔display curves. No colour, canvas, DOM, or defringe knowledge.

export const DEG = Math.PI / 180;
export const TAU = 2 * Math.PI;

const RESPONSE_GAMMA = 0.6;   // <1 = concave: low/default values get a generous radius, big ranges still fit

export function clamp01(x) { return Math.max(0, Math.min(1, x)); }
export function lerp(a, b, t) { return a + (b - a) * t; }
export function safeSpan(span) { return Math.max(span, 1e-6); }
export function roundTo(value, decimals) { const factor = Math.pow(10, decimals); return Math.round(value * factor) / factor; }

export function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
export function polar(center, angleRad, radius) {
  return { x: center.x + radius * Math.cos(angleRad), y: center.y - radius * Math.sin(angleRad) };
}
export function unitVectorFrom(center, point) {
  const len = distance(point, center) || 1;
  return { x: (point.x - center.x) / len, y: (point.y - center.y) / len };
}
export function bearing(point, center) { return Math.atan2(-(point.y - center.y), point.x - center.x); }   // radians, screen y-down corrected
export function angleAt(point, center) { return bearing(point, center) / DEG; }                            // the same, in degrees
export function wrapSigned(deg) { return ((deg + 180) % 360 + 360) % 360 - 180; }                           // -> (-180, 180]
export function chordHalfWidth(radius, offsetFromCenter) { return Math.sqrt(Math.max(radius * radius - offsetFromCenter ** 2, 0)); }
export function verticalFraction(y, centerY, radius) { return (centerY + radius - y) / (2 * radius); }      // top = 1, bottom = 0

export function snapToRange(value, range) {
  const stepped = Math.round(value / range.step) * range.step;
  return Math.max(range.min, Math.min(range.max, parseFloat(stepped.toFixed(4))));
}
export function growRadius(value, valueMax, radiusMax) {
  return radiusMax * Math.pow(Math.max(0, value) / safeSpan(valueMax), RESPONSE_GAMMA);
}
export function shrinkToValue(px, valueMax, radiusMax) {
  return safeSpan(valueMax) * Math.pow(Math.max(0, px) / safeSpan(radiusMax), 1 / RESPONSE_GAMMA);
}
