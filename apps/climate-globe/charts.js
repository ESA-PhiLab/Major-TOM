// Small line charts for the selected cell, drawn as SVG: the line in Major TOM blue, an optional band (e.g. ±1
// standard deviation) behind it, the shown month marked in orange; months without a value leave a gap.

const W = 308, H = 132, M = { left: 38, right: 8, top: 8, bottom: 20 }
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

/** Draw into `svg`: y[i] at x[i] (x in data units, xRange [min, max]); optional band lo[i]..hi[i];
 *  xTicks [[x, label]]; marker = index to mark, or -1. */
export function lineChart(svg, { x, y, lo, hi, xRange, xTicks, marker = -1 }) {
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`)
  svg.replaceChildren()
  const values = [...y, ...(lo ?? []), ...(hi ?? [])].filter(Number.isFinite)
  if (!values.length) { svg.append(el('text', { x: W / 2, y: H / 2, class: 'empty' }, 'no values')); return }
  const yt = ticks(Math.min(0, ...values), Math.max(...values))
  const y0 = Math.min(yt[0], ...values), y1 = Math.max(yt.at(-1), ...values)
  const sx = v => M.left + ((v - xRange[0]) / (xRange[1] - xRange[0])) * (W - M.left - M.right)
  const sy = v => H - M.bottom - ((v - y0) / (y1 - y0 || 1)) * (H - M.top - M.bottom)

  for (const t of yt) {
    svg.append(el('line', { x1: M.left, x2: W - M.right, y1: sy(t), y2: sy(t), class: 'grid' }))
    svg.append(el('text', { x: M.left - 5, y: sy(t) + 3, class: 'tick', 'text-anchor': 'end' }, label(t)))
  }
  for (const [t, name] of xTicks) svg.append(el('text', { x: sx(t), y: H - 5, class: 'tick', 'text-anchor': 'middle' }, name))

  /** Path through the finite points, broken where a value is missing. */
  const path = ys => ys.map((v, i) => (Number.isFinite(v) ? `${i && Number.isFinite(ys[i - 1]) ? 'L' : 'M'}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}` : '')).join('')
  if (lo && hi) {
    for (let i = 0; i < x.length;) {                                    // one band piece per run of finite values
      if (!Number.isFinite(lo[i]) || !Number.isFinite(hi[i])) { i++; continue }
      let j = i
      while (j < x.length && Number.isFinite(lo[j]) && Number.isFinite(hi[j])) j++
      const top = hi.slice(i, j).map((v, k) => `${sx(x[i + k]).toFixed(1)},${sy(v).toFixed(1)}`)
      const bottom = lo.slice(i, j).map((v, k) => `${sx(x[i + k]).toFixed(1)},${sy(v).toFixed(1)}`).reverse()
      svg.append(el('polygon', { points: [...top, ...bottom].join(' '), class: 'band' }))
      i = j
    }
  }
  svg.append(el('path', { d: path(y), class: 'line' }))
  if (marker >= 0 && Number.isFinite(y[marker])) {
    svg.append(el('circle', { cx: sx(x[marker]), cy: sy(y[marker]), r: 3.5, class: 'marker' }))
  }
}
