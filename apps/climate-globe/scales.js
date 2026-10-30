// Colour scales: the README figures' colour ramps (docs/evidence/climate_figures.py), as 256-step lookup tables.
// A value -> position 0..1 on the scale (log, linear, or two slopes around a centre) -> one of 256 colours.
// Values beyond the ends take the end colours (a log scale has no place for 0 mm: a month without rain is driest).

export const RAMPS = {
  dryWet: ['#f2e3b6', '#cfdc8f', '#8fc77d', '#47ad93', '#2f8fc9', '#4a6ad6'],
  coldHot: ['#2c66c8', '#5b8ee0', '#9dbbea', '#dddbd5', '#eeaa9f', '#e0675a', '#c8372f'],
  haze: ['#f6e6c8', '#f2c784', '#eba04f', '#d9742c', '#b9521f'],
}

const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16))

/** 256 RGB colours, evenly interpolated between the ramp's stops (as matplotlib's from_list). */
function lookupTable(stops) {
  const c = stops.map(rgb), lut = new Uint8Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (c.length - 1), k = Math.min(Math.floor(x), c.length - 2), f = x - k
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = Math.round(c[k][j] + (c[k + 1][j] - c[k][j]) * f)
  }
  return lut
}

/** A scale: {type: 'log'|'linear'|'centred', min, max, centre?, ramp, ticks} -> position(v), lut. */
export function makeScale(spec) {
  const { type, min, max, centre = 0 } = spec
  const position =
    type === 'log' ? v => (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min))
    : type === 'centred' ? v => (v < centre ? 0.5 * (v - min) / (centre - min) : 0.5 + 0.5 * (v - centre) / (max - centre))
    : v => (v - min) / (max - min)
  return { ...spec, lut: lookupTable(RAMPS[spec.ramp]), position: v => Math.min(1, Math.max(0, position(Math.max(v, type === 'log' ? min : -Infinity)))) }
}

/** Draw the scale as a horizontal bar into a canvas (legend). */
export function drawBar(canvas, scale) {
  const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height
  for (let x = 0; x < w; x++) {
    const i = Math.round((x / (w - 1)) * 255) * 3
    ctx.fillStyle = `rgb(${scale.lut[i]},${scale.lut[i + 1]},${scale.lut[i + 2]})`
    ctx.fillRect(x, 0, 1, h)
  }
}
