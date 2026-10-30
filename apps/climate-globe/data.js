// Where a layer's values come from, with the bytes and time each load costs.
//
//   preview:  previews/<dataset>/<variable>/MM.png (climatology, colours only)  -> fetch -> decode -> RGBA pixels
//   live:     <source.coop>/<dataset>/series/year=YYYY/MM.parquet, 3 columns     -> range requests (hyparquet)
//             -> grid_row_u, grid_col_r, value per cell -> Float32Array on the (row, column) array

import { parquetMetadataAsync, parquetRead } from 'https://cdn.jsdelivr.net/npm/hyparquet@1.31.4/+esm'
import { compressors } from 'https://cdn.jsdelivr.net/npm/hyparquet-compressors@1.1.2/+esm'
import { COLS, ROWS } from './tiles.js'

export const SOURCE_COOP = 'https://data.source.coop/major-tom/climate'
const HOURS = [744, 678, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]        // Feb: 28.25 days

export const LAYERS = [
  { id: 'cpc-precip', name: 'Precipitation', source: 'CPC, rain gauges', dataset: 'CPC-Monthly', column: 'cpc_precip',
    unit: 'mm per month', scale: { type: 'log', min: 4, max: 400, ramp: 'dryWet', ticks: [4, 10, 25, 50, 100, 200, 400] },
    preview: 'images/CPC-Monthly/precip/{MM}.png', series: [1979, 2025] },
  { id: 'imerg-precip', name: 'Precipitation', source: 'IMERG, satellites', dataset: 'IMERG-Monthly', column: 'imerg_precip',
    unit: 'mm per month', perMonth: m => HOURS[m - 1],                            // stored as mm/h
    scale: { type: 'log', min: 4, max: 400, ramp: 'dryWet', ticks: [4, 10, 25, 50, 100, 200, 400] }, series: [1998, 2024] },
  { id: 'cpc-tmax', name: 'Daily maximum temperature', source: 'CPC, stations', dataset: 'CPC-Monthly', column: 'cpc_tmax',
    unit: '°C', scale: { type: 'centred', min: -40, centre: 0, max: 40, ramp: 'coldHot', ticks: [-40, -20, 0, 20, 40] },
    series: [1979, 2025] },
  { id: 'cams-pm25', name: 'Surface PM2.5', source: 'CAMS, reanalysis', dataset: 'CAMS-Monthly', column: 'cams_pm2p5',
    unit: 'µg/m³', perMonth: () => 1e9,                                          // stored as kg/m³
    scale: { type: 'log', min: 2, max: 150, ramp: 'haze', ticks: [2, 5, 10, 25, 50, 150] }, series: [2003, 2025] },
]

/** An AsyncBuffer for hyparquet over HTTP range requests that counts the bytes it downloads. */
async function countedFile(url, stats) {
  const head = await fetch(url, { method: 'HEAD' })
  if (!head.ok) throw new Error(`${url}: HTTP ${head.status}`)
  const byteLength = Number(head.headers.get('content-length'))
  return {
    byteLength,
    async slice(start, end = byteLength) {
      if (stats.active++ === 0) stats.since = performance.now()               // network time: any request open
      const r = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } })
      const buffer = await r.arrayBuffer()
      if (--stats.active === 0) stats.network += performance.now() - stats.since
      stats.bytes += buffer.byteLength; stats.requests += 1
      return buffer
    },
  }
}

const PARALLEL = 16             // requests open at once: each range request waits ~0.2-1.3 s before its first byte

/** Columns of a remote Parquet file as typed arrays, null -> NaN. One read per row group, PARALLEL at a time:
 *  hyparquet reads the row groups of one call one after another, which spends the time waiting on requests. */
async function readColumns(url, columns, stats) {
  const file = await countedFile(url, stats)
  const metadata = await parquetMetadataAsync(file)
  const n = Number(metadata.num_rows), out = Object.fromEntries(columns.map(([name, Type]) => [name, new Type(n)]))
  const queue = []
  for (let start = 0, g = 0; g < metadata.row_groups.length; g++) {
    const end = start + Number(metadata.row_groups[g].num_rows)
    queue.push([start, end]); start = end
  }
  await Promise.all(Array.from({ length: PARALLEL }, async () => {
    for (let r; (r = queue.shift());) {
      await parquetRead({
        file, metadata, compressors, columns: Object.keys(out), rowStart: r[0], rowEnd: r[1],
        onChunk({ columnName, columnData, rowStart }) {
          const target = out[columnName]
          if (typeof columnData[0] === 'bigint') {                // 64-bit integers (cells.parquet positions)
            for (let i = 0; i < columnData.length; i++) target[rowStart + i] = Number(columnData[i])
          } else {
            for (let i = 0; i < columnData.length; i++) target[rowStart + i] = columnData[i] ?? NaN
          }
        },
      })
    }
  }))
  return out
}

/** Every dataset file lists its cells in the order of its cells.parquet (the release layout, majortom/climate/
 *  layout.py), so positions are read once per dataset: index into the (row, column) array of each row. */
const positions = new Map()
function cellPositions(dataset, stats) {
  if (!positions.has(dataset)) {
    positions.set(dataset, readColumns(`${SOURCE_COOP}/${dataset}/cells.parquet`,
      [['grid_row_u', Int16Array], ['grid_col_r', Int16Array]], stats).then(({ grid_row_u: r, grid_col_r: c }) => {
      const index = new Int32Array(r.length)
      for (let i = 0; i < r.length; i++) index[i] = (r[i] + ROWS / 2) * COLS + c[i] + COLS / 2
      return index
    }))
  }
  return positions.get(dataset)
}

/** One month of a layer's series, read live: values on the (row, column) array, NaN where missing. */
export async function loadSeries(layer, year, month) {
  const stats = { bytes: 0, requests: 0, network: 0, active: 0, t0: performance.now() }
  const url = `${SOURCE_COOP}/${layer.dataset}/series/year=${year}/${String(month).padStart(2, '0')}.parquet`
  stats.firstOfDataset = !positions.has(layer.dataset)
  const [index, cols] = await Promise.all([cellPositions(layer.dataset, stats),
    readColumns(url, [[layer.column, Float32Array]], stats)])
  stats.read = performance.now() - stats.t0
  const v = cols[layer.column]
  if (v.length !== index.length) throw new Error(`${url}: ${v.length} rows, cells.parquet has ${index.length}`)
  const values = new Float32Array(ROWS * COLS).fill(NaN), factor = layer.perMonth ? layer.perMonth(month) : 1
  for (let i = 0; i < v.length; i++) values[index[i]] = v[i] * factor
  stats.cells = v.length
  return { values, stats }
}

/** One month of a layer's climatology preview image: RGBA pixels, width x width/2. */
export async function loadPreview(layer, month) {
  const stats = { bytes: 0, requests: 1, t0: performance.now() }
  const blob = await (await fetch(layer.preview.replace('{MM}', String(month).padStart(2, '0')))).blob()
  stats.bytes = blob.size
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })  // colours as stored
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height), ctx = canvas.getContext('2d')
  ctx.drawImage(bitmap, 0, 0)
  stats.read = performance.now() - stats.t0
  return { rgba: ctx.getImageData(0, 0, bitmap.width, bitmap.height).data, width: bitmap.width, stats }
}
