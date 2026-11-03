// Core-Weather for one sample, from source.coop: the hourly (±24 h) and daily (±30 d) rows of its acquisition.
//
//   sample (cell name, timestamp "20221211T114501") -> <hourly|daily>/year=YYYY/YYYY-MM-DD.parquet (one row group,
//     rows sorted by grid_cell) -> footer, grid_cell and acquired (whole column, ~20 KB) -> the sample's row range
//     -> the chart columns for that range only -> one small chart per variable, step 0 marked
//   ERA5 for temperature, precipitation and cloud: it covers every cell (ERA5-Land is land only, and south of 57.1°S
//   is not filled yet).

import { compressors, newStats, openFile, parquetRead } from './parquet.js'

export const WEATHER = 'https://data.source.coop/major-tom/core-weather'

// [column, label, source, unit, value as shown]
const HOURLY = [
  ['era5_t2m', 'Temperature', 'ERA5, 2 m', '°C', v => v - 273.15],
  ['era5_tp', 'Precipitation', 'ERA5', 'mm', v => v * 1000],
  ['era5_tcc', 'Cloud cover', 'ERA5', '%', v => v * 100],
]
const DAILY = [
  ['era5_t2m_mean', 'Temperature', 'ERA5, 2 m', '°C', v => v - 273.15],
  ['era5_tp_sum', 'Precipitation', 'ERA5', 'mm', v => v * 1000],
  ['cams_pm2p5_mean', 'PM2.5', 'CAMS', 'µg/m³', v => v * 1e9],
  ['s5p_no2', 'NO₂ column', 'Sentinel-5P', 'µmol/m²', v => v * 1e6],
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

/** One row per variable: label, a line over the steps with step 0 marked, the value at step 0. */
export function weatherRows(table, which) {
  if (!table) return []
  const spec = which === 'hourly' ? HOURLY : DAILY
  return spec.map(([column, label, source, unit, show]) => {
    const values = table[column].map(v => (v === null || Number.isNaN(v) ? null : show(v)))
    const at0 = values[table.step.indexOf(0)]
    const row = document.createElement('div')
    row.className = 'wx-row'
    row.innerHTML = `<span class="wx-label">${label}<small>${source}</small></span>${spark(table.step, values)}` +
      `<span class="wx-value">${at0 === null || at0 === undefined ? '–' : `${fmt(at0)} ${unit}`}</span>`
    return row
  })
}

const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2))

/** An inline SVG line (gaps where values are missing) with a dot at step 0; the y range is the series' own. */
function spark(steps, values) {
  const W = 120, H = 28, P = 3
  const known = values.filter(v => v !== null)
  if (!known.length) return `<svg class="wx-spark" viewBox="0 0 ${W} ${H}"></svg>`
  const lo = Math.min(...known), hi = Math.max(...known), s0 = steps[0], s1 = steps[steps.length - 1]
  const x = s => P + ((s - s0) / (s1 - s0 || 1)) * (W - 2 * P)
  const y = v => (hi === lo ? H / 2 : H - P - ((v - lo) / (hi - lo)) * (H - 2 * P))
  let d = '', pen = false
  steps.forEach((s, i) => {
    if (values[i] === null) { pen = false; return }
    d += `${pen ? 'L' : 'M'}${x(s).toFixed(1)},${y(values[i]).toFixed(1)}`; pen = true
  })
  const i0 = steps.indexOf(0), dot = i0 >= 0 && values[i0] !== null
    ? `<circle cx="${x(0).toFixed(1)}" cy="${y(values[i0]).toFixed(1)}" r="2.5" class="wx-now"/>` : ''
  return `<svg class="wx-spark" viewBox="0 0 ${W} ${H}" aria-hidden="true">` +
    `<line x1="${x(0)}" x2="${x(0)}" y1="0" y2="${H}" class="wx-zero"/><path d="${d}"/>${dot}</svg>`
}
