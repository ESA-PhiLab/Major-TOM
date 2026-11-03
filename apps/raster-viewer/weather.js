// Core-Weather for one sample, from source.coop: the hourly (±24 h) and daily (±30 d) rows of its acquisition.
//
//   sample (cell name, timestamp "20221211T114501") -> <hourly|daily>/year=YYYY/YYYY-MM-DD.parquet (one row group,
//     rows sorted by grid_cell) -> footer, grid_cell and acquired (whole column, ~20 KB) -> the sample's row range
//     -> the chart columns for that range only -> one small chart per variable, step 0 marked
//   ERA5 for temperature, precipitation and cloud: it covers every cell (ERA5-Land is land only, and south of 57.1°S
//   is not filled yet).

import { compressors, newStats, openFile, parquetRead } from './parquet.js'

export const WEATHER = 'https://data.source.coop/major-tom/core-weather'

// [column, label, source, unit, value as shown, mark]: line (states), bars from zero (amounts), dots (one overpass a day)
const HOURLY = [
  ['era5_t2m', 'Temperature', 'ERA5, 2 m', '°C', v => v - 273.15, 'line'],
  ['era5_tp', 'Precipitation', 'ERA5', 'mm', v => v * 1000, 'bars'],
  ['era5_tcc', 'Cloud cover', 'ERA5', '%', v => v * 100, 'line'],
]
const DAILY = [
  ['era5_t2m_mean', 'Temperature', 'ERA5, 2 m, daily mean', '°C', v => v - 273.15, 'line'],
  ['era5_tp_sum', 'Precipitation', 'ERA5', 'mm', v => v * 1000, 'bars'],
  ['cams_pm2p5_mean', 'PM2.5', 'CAMS', 'µg/m³', v => v * 1e9, 'line'],
  ['s5p_no2', 'NO₂ column', 'Sentinel-5P', 'µmol/m²', v => v * 1e6, 'dots'],
]

const ms = v => (v instanceof Date ? v.getTime() : typeof v === 'bigint' ? Number(v / 1000000n) : Number(v))

/** The rows of one table for (cell, acquired ms): {step: [...], <column>: [...]}, or null if the cell is absent. */
async function readTable(table, day, cell, acquired, columns, step, stats) {
  const url = `${WEATHER}/${table}/year=${day.slice(0, 4)}/${day}.parquet`
  const { file, metadata } = await openFile(url, stats)
  let keys
  await parquetRead({ file, metadata, compressors, columns: ['grid_cell', 'acquired'], rowFormat: 'object',
    onComplete(rows) { keys = rows } })
  let rowStart = -1, rowEnd = -1
  for (let i = 0; i < keys.length; i++) {
    if (keys[i].grid_cell !== cell || ms(keys[i].acquired) !== acquired) { if (rowStart >= 0) break; continue }
    if (rowStart < 0) rowStart = i
    rowEnd = i + 1
  }
  if (rowStart < 0) return null
  let rows
  await parquetRead({ file, metadata, compressors, columns: [step, ...columns], rowStart, rowEnd, rowFormat: 'object',
    onComplete(r) { rows = r } })
  const out = { step: rows.map(r => Number(r[step])) }
  for (const c of columns) out[c] = rows.map(r => (r[c] === null || r[c] === undefined ? null : Number(r[c])))
  return out
}

/** Both tables for one sample: {hourly, daily, stats}; a table is null where Core-Weather has no row. */
export async function loadWeather(cell, timestamp) {
  const stats = newStats()
  const t = timestamp, day = `${t.slice(0, 4)}-${t.slice(4, 6)}-${t.slice(6, 8)}`
  const acquired = Date.UTC(+t.slice(0, 4), +t.slice(4, 6) - 1, +t.slice(6, 8), +t.slice(9, 11), +t.slice(11, 13), +t.slice(13, 15))
  const [hourly, daily] = await Promise.all([
    readTable('hourly', day, cell, acquired, HOURLY.map(c => c[0]), 'step_h', stats),
    readTable('daily', day, cell, acquired, DAILY.map(c => c[0]), 'step_d', stats),
  ])
  stats.read = performance.now() - stats.t0
  return { hourly, daily, stats }
}

/** One block per variable: name, source and the value at the acquisition above a chart of the steps; then the
 *  steps' axis. */
export function weatherRows(table, which) {
  if (!table) return []
  const spec = which === 'hourly' ? HOURLY : DAILY, unit = which === 'hourly' ? 'h' : 'd'
  const blocks = spec.map(([column, label, source, units, show, mark]) => {
    const values = table[column].map(v => (v === null || Number.isNaN(v) ? null : show(v)))
    const at0 = values[table.step.indexOf(0)]
    const block = document.createElement('div')
    block.className = 'wx-row'
    block.innerHTML = `<div class="wx-top"><span class="wx-label">${label} <small>${source}</small></span>` +
      `<span class="wx-value">${at0 === null || at0 === undefined ? '–' : `${fmt(at0)}<small> ${units}</small>`}</span></div>` +
      chart(table.step, values, mark)
    return block
  })
  const s0 = table.step[0], s1 = table.step[table.step.length - 1]
  const axis = document.createElement('div')
  axis.className = 'wx-axis'
  axis.innerHTML = `<span>${s0} ${unit}</span><span>acquisition</span><span>+${s1} ${unit}</span>`
  return [...blocks, axis]
}

const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2))

/** A chart of one series over the steps: its own y range (from zero for bars), the acquisition as a vertical line
 *  with a dot, the range's ends written at the right. Drawn at 300 x 44 and scaled as a whole (no distortion). */
function chart(steps, values, mark) {
  const W = 300, H = 44, L = 2, R = 40, T = 5, B = 5
  const known = values.filter(v => v !== null)
  const empty = `<svg class="wx-chart" viewBox="0 0 ${W} ${H}" aria-hidden="true"><text x="${(W - R) / 2}" y="${H / 2 + 4}" class="wx-none">no data</text></svg>`
  if (!known.length) return empty
  let lo = Math.min(...known), hi = Math.max(...known)
  if (mark === 'bars') lo = 0
  if (hi === lo) hi = lo + 1
  const s0 = steps[0], s1 = steps[steps.length - 1], step = (W - L - R) / (s1 - s0)
  const x = s => L + (s - s0) * step
  const y = v => H - B - ((v - lo) / (hi - lo)) * (H - T - B)
  let body = ''
  if (mark === 'bars') {
    const w = Math.max(step * 0.7, 1)
    steps.forEach((s, i) => {
      if (values[i] > 0) body += `<rect x="${(x(s) - w / 2).toFixed(1)}" y="${y(values[i]).toFixed(1)}" width="${w.toFixed(1)}" height="${(y(0) - y(values[i])).toFixed(1)}" rx="1"/>`
    })
    body = `<line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" class="wx-base"/><g class="wx-bars">${body}</g>`
  } else if (mark === 'dots') {
    steps.forEach((s, i) => { if (values[i] !== null) body += `<circle cx="${x(s).toFixed(1)}" cy="${y(values[i]).toFixed(1)}" r="2"/>` })
    body = `<g class="wx-dots">${body}</g>`
  } else {
    const runs = [[]]
    steps.forEach((s, i) => (values[i] === null ? runs.push([]) : runs[runs.length - 1].push([x(s), y(values[i])])))
    for (const run of runs.filter(r => r.length)) {
      const line = run.map(([a, b], k) => `${k ? 'L' : 'M'}${a.toFixed(1)},${b.toFixed(1)}`).join('')
      body += `<path class="wx-area" d="${line}L${run[run.length - 1][0].toFixed(1)},${H - B}L${run[0][0].toFixed(1)},${H - B}Z"/>` +
        `<path class="wx-line" d="${line}"/>`
    }
  }
  const i0 = steps.indexOf(0), v0 = values[i0]
  const now = `<line x1="${x(0)}" x2="${x(0)}" y1="${T - 3}" y2="${H - B}" class="wx-zero"/>` +
    (v0 !== null && v0 !== undefined ? `<circle cx="${x(0)}" cy="${y(v0).toFixed(1)}" r="3.5" class="wx-now"/>` : '')
  const ends = `<text x="${W}" y="${T + 6}" class="wx-tick">${fmt(hi)}</text><text x="${W}" y="${H - B}" class="wx-tick">${fmt(lo)}</text>`
  return `<svg class="wx-chart" viewBox="0 0 ${W} ${H}" aria-hidden="true">${body}${now}${ends}</svg>`
}
