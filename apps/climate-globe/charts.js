// Small line charts for the selected cell, drawn as SVG: the line in Major TOM blue, an optional band (e.g. ±1
// standard deviation) behind it, the shown month marked in orange; months without a value leave a gap.
// The y-axis starts at 0 only for amounts (all values >= 0 and the smallest under half the largest); values in
// kelvin also get °C labels on the right, on the same axis. A window (xRange narrower than the data) is clipped.

const W = 308, H = 132, M = { left: 38, right: 8, top: 8, bottom: 20 }, C_RIGHT = 30
export const KELVIN = 273.15
let clipId = 0
const NS = 'http://www.w3.org/2000/svg'

function el(name, attrs, text) {
  const e = document.createElementNS(NS, name)
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v)
  if (text !== undefined) e.textContent = text
  return e
}

/** About `n` round tick values covering lo..hi. */
function ticks(lo, hi, n = 3) {
  const raw = (hi - lo) / n || 1, p = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map(f => f * p).find(s => s >= raw)
  const out = []
  for (let t = Math.floor(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(+t.toFixed(10))
  return out
}

const label = v => (Math.abs(v) >= 1000 ? `${v / 1000}k` : `${v}`)

/** The share of the chart's width that is plot area (for turning a drag in pixels into data units). */
export const plotShare = celsius => (W - M.left - M.right - (celsius ? C_RIGHT : 0)) / W

/** Draw into `svg`: y[i] at x[i] (x in data units, xRange [min, max], points outside it clipped); optional band
 *  lo[i]..hi[i]; xTicks [[x, label]]; marker = index to mark, or -1; yAll: the values the y-axis is fitted to
 *  (default: y and the band; pass the whole series so the axis stays put while a window moves); celsius: also
 *  label the axis in °C (values in K). */
export function lineChart(svg, { x, y, lo, hi, xRange, xTicks, marker = -1, yAll, celsius = false }) {
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`)
  svg.replaceChildren()
  const values = (yAll ?? [...y, ...(lo ?? []), ...(hi ?? [])]).filter(Number.isFinite)
  if (!values.length) { svg.append(el('text', { x: W / 2, y: H / 2, class: 'empty' }, 'no values')); return }
  let min = Math.min(...values), max = Math.max(...values)
  if (min >= 0 && min < max / 2) min = 0                                  // amounts: from zero
  if (min === max) { min -= 1; max += 1 }
  const pad = (max - min) * 0.04                                           // the axis fits the values
  const y0 = min === 0 ? 0 : min - pad, y1 = max + pad
  const yt = ticks(y0, y1).filter(t => t >= y0 && t <= y1)
  const right = W - M.right - (celsius ? C_RIGHT : 0)
  const sx = v => M.left + ((v - xRange[0]) / (xRange[1] - xRange[0])) * (right - M.left)
  const sy = v => H - M.bottom - ((v - y0) / (y1 - y0 || 1)) * (H - M.top - M.bottom)

  for (const t of yt) {
    svg.append(el('line', { x1: M.left, x2: right, y1: sy(t), y2: sy(t), class: 'grid' }))
    svg.append(el('text', { x: M.left - 5, y: sy(t) + 3, class: 'tick', 'text-anchor': 'end' }, label(t)))
  }
  if (celsius) {                                                          // the same axis in °C: own round values
    for (const c of ticks(y0 - KELVIN, y1 - KELVIN)) {
      if (c + KELVIN < y0 || c + KELVIN > y1) continue
      svg.append(el('line', { x1: right, x2: right + 3, y1: sy(c + KELVIN), y2: sy(c + KELVIN), class: 'grid' }))
      svg.append(el('text', { x: right + 5, y: sy(c + KELVIN) + 3, class: 'tick' }, `${label(c)}°C`))
    }
  }
  const clip = `plot-${++clipId}`                                          // a window: nothing drawn past its edges
  const clipPath = el('clipPath', { id: clip })
  clipPath.append(el('rect', { x: M.left, y: 0, width: right - M.left, height: H }))
  svg.append(clipPath)
  const plot = el('g', { 'clip-path': `url(#${clip})` })
  for (const [t, name] of xTicks) if (t >= xRange[0] && t <= xRange[1]) svg.append(el('text', { x: sx(t), y: H - 5, class: 'tick', 'text-anchor': 'middle' }, name))

  /** Path through the finite points, broken where a value is missing. */
  const path = ys => ys.map((v, i) => (Number.isFinite(v) ? `${i && Number.isFinite(ys[i - 1]) ? 'L' : 'M'}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}` : '')).join('')
  if (lo && hi) {
    for (let i = 0; i < x.length;) {                                    // one band piece per run of finite values
      if (!Number.isFinite(lo[i]) || !Number.isFinite(hi[i])) { i++; continue }
      let j = i
      while (j < x.length && Number.isFinite(lo[j]) && Number.isFinite(hi[j])) j++
      const top = hi.slice(i, j).map((v, k) => `${sx(x[i + k]).toFixed(1)},${sy(v).toFixed(1)}`)
      const bottom = lo.slice(i, j).map((v, k) => `${sx(x[i + k]).toFixed(1)},${sy(v).toFixed(1)}`).reverse()
      plot.append(el('polygon', { points: [...top, ...bottom].join(' '), class: 'band' }))
      i = j
    }
  }
  plot.append(el('path', { d: path(y), class: 'line' }))
  svg.append(plot)
  if (marker >= 0 && Number.isFinite(y[marker]) && x[marker] >= xRange[0] && x[marker] <= xRange[1]) {
    svg.append(el('circle', { cx: sx(x[marker]), cy: sy(y[marker]), r: 3.5, class: 'marker' }))
  }
}
