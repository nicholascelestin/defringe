// CIELAB ↔ sRGB and the a*b* chromaticity disc as pixel data. No canvas element, no defringe.

import { clamp01 } from './geometry.js';

function labToRgb(L, a, b) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const f = (t) => (t > 6 / 29 ? t * t * t : 3 * (6 / 29) * (6 / 29) * (t - 4 / 29));
  const X = 0.95047 * f(fx), Y = f(fy), Z = 1.08883 * f(fz);            // D65
  const linear = [3.2406 * X - 1.5372 * Y - 0.4986 * Z,
                  -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
                  0.0557 * X - 0.2040 * Y + 1.0570 * Z];
  return linear.map((channel) => {
    channel = channel <= 0.0031308 ? 12.92 * channel : 1.055 * Math.pow(Math.max(0, channel), 1 / 2.4) - 0.055;
    return clamp01(channel) * 255;
  });
}

export function hexToRgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${alpha})`;
}

// The CIELAB a*b* plane at a fixed lightness, as a square bitmap: angle = hue, radius = chroma,
// transparent outside the disc. Pixel sizes in, pixel data out — dpr and canvas are the caller's concern.
export function abDiscImageData(diameterPx, radiusPx, maxChroma, lightness) {
  const image = new ImageData(diameterPx, diameterPx), pixels = image.data, center = diameterPx / 2;
  for (let y = 0; y < diameterPx; y++) {
    for (let x = 0; x < diameterPx; x++) {
      const dx = x - center, dy = y - center, i = (y * diameterPx + x) * 4;
      if (dx * dx + dy * dy > radiusPx * radiusPx) continue;            // outside the disc -> transparent
      const rgb = labToRgb(lightness, maxChroma * dx / radiusPx, -maxChroma * dy / radiusPx);   // screen y is down
      pixels[i] = rgb[0]; pixels[i + 1] = rgb[1]; pixels[i + 2] = rgb[2]; pixels[i + 3] = 255;
    }
  }
  return image;
}
