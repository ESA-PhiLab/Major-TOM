// Where a layer's values come from, with the bytes and time each load costs.
//
//   preview:  previews/<dataset>/<variable>/MM.png (climatology, colours only)  -> fetch -> decode -> RGBA pixels
//   live:     <source.coop>/<dataset>/series/year=YYYY/MM.parquet, 3 columns     -> range requests (hyparquet)
//             -> grid_row_u, grid_col_r, value per cell -> Float32Array on the (row, column) array

import { parquetMetadataAsync, parquetRead } from 'https://cdn.jsdelivr.net/npm/hyparquet@1.31.4/+esm'
import { compressors } from 'https://cdn.jsdelivr.net/npm/hyparquet-compressors@1.1.2/+esm'
import * as zarr from 'https://cdn.jsdelivr.net/npm/zarrita@0.7.5/+esm'
import { COLS, ROWS } from './tiles.js'

export const SOURCE_COOP = 'https://data.source.coop/major-tom/climate'
const HOURS = [744, 678, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]        // Feb: 28.25 days

export const LAYERS = [
  { id: 'cpc-precip', name: 'Precipitation', source: 'CPC, rain gauges', dataset: 'CPC-Monthly', column: 'cpc_precip',
    unit: 'mm per month', scale: { type: 'log', min: 4, max: 400, ramp: 'dryWet', ticks: [4, 10, 25, 50, 100, 200, 400] },
    preview: 'images/CPC-Monthly/precip/{MM}.png', series: [1979, 2025], normal: '1991–2020', tables: true,
    zarr: { path: '_experiments/cpc-precip.zarr', years: [2024, 2025] } },               // test store, Zarr comparison
  { id: 'imerg-precip', name: 'Precipitation', source: 'IMERG, satellites', dataset: 'IMERG-Monthly', column: 'imerg_precip',
    unit: 'mm per month', perMonth: m => HOURS[m - 1],                            // stored as mm/h
    scale: { type: 'log', min: 4, max: 400, ramp: 'dryWet', ticks: [4, 10, 25, 50, 100, 200, 400] },
    preview: 'images/IMERG-Monthly/precip/{MM}.png', series: [1998, 2024], normal: '1998–2024', tables: true },
  { id: 'cpc-tmax', name: 'Daily maximum temperature', source: 'CPC, stations', dataset: 'CPC-Monthly', column: 'cpc_tmax',
    unit: '°C', scale: { type: 'centred', min: -40, centre: 0, max: 40, ramp: 'coldHot', ticks: [-40, -20, 0, 20, 40] },
    preview: 'images/CPC-Monthly/tmax/{MM}.png',
    series: [1979, 2025], normal: '1991–2020', tables: true },
  { id: 'cams-pm25', name: 'Surface PM2.5', source: 'CAMS, reanalysis', dataset: 'CAMS-Monthly', column: 'cams_pm2p5',
    unit: 'µg/m³', perMonth: () => 1e9,                                          // stored as kg/m³
    scale: { type: 'log', min: 2, max: 150, ramp: 'haze', ticks: [2, 5, 10, 25, 50, 150] },
    preview: 'images/CAMS-Monthly/pm25/{MM}.png', series: [2003, 2025], normal: '2003–2025', tables: true },
]

const TAIL = 131072             // a file's first request: its last 128 KB, which hold the footer (28-97 KB here);
                                // hyparquet is told the same size, or it fetches its own last 512 KB again

/** One range request, counted: bytes, requests, and network time (while any request is open). */
async function rangeGet(url, range, stats) {
  if (stats.active++ === 0) stats.since = performance.now()
  try {
    const r = await fetch(url, { headers: { Range: `bytes=${range}` } })
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`)
    const buffer = await r.arrayBuffer()
    stats.bytes += buffer.byteLength; stats.requests += 1
    return [r, buffer]
  } finally {
    if (--stats.active === 0) stats.network += performance.now() - stats.since
  }
}

/** An AsyncBuffer for hyparquet over HTTP range requests. Opening it reads the file's tail, which gives the file
 *  size (Content-Range) and the footer in one request, where a size request and a footer request would take two. */
async function countedFile(url, stats, tailSize = TAIL) {
  const [r, tail] = await rangeGet(url, `-${tailSize}`, stats)
  const range = r.headers.get('content-range')                                    // "bytes a-b/size"; absent if whole
  const byteLength = range ? Number(range.split('/')[1]) : tail.byteLength
  const tailStart = byteLength - tail.byteLength
  return {
    byteLength,
    async slice(start, end = byteLength) {
      if (start >= tailStart) return tail.slice(start - tailStart, end - tailStart)
      return (await rangeGet(url, `${start}-${end - 1}`, stats))[1]
    },
  }
}

const PARALLEL = 16             // requests open at once: each range request waits ~0.2-1.3 s before its first byte

/** Columns of a remote Parquet file as typed arrays, null -> NaN. One read per row group, PARALLEL at a time:
 *  hyparquet reads the row groups of one call one after another, which spends the time waiting on requests. */
async function readColumns(url, columns, stats, progress) {
  const file = await countedFile(url, stats)
  const metadata = await parquetMetadataAsync(file, { initialFetchSize: TAIL })
  progress.total += metadata.row_groups.length
  for (const group of metadata.row_groups) {                                       // bytes the columns will take
    for (const { meta_data: m } of group.columns) {
      if (columns.some(([name]) => name === m.path_in_schema.join('.'))) progress.expected += Number(m.total_compressed_size)
    }
  }
  progress.report()
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
      progress.done += 1; progress.report()
    }
  }))
  return out
}

/** Every dataset file lists its cells in the order of its cells.parquet (the release layout, majortom/climate/
 *  layout.py), so positions are read once per dataset: index into the (row, column) array of each row. */
const positions = new Map()
function cellPositions(dataset, stats, progress) {
  if (!positions.has(dataset)) {
    positions.set(dataset, readColumns(`${SOURCE_COOP}/${dataset}/cells.parquet`,
      [['grid_row_u', Int16Array], ['grid_col_r', Int16Array]], stats, progress).then(({ grid_row_u: r, grid_col_r: c }) => {
      const index = new Int32Array(r.length)
      for (let i = 0; i < r.length; i++) index[i] = (r[i] + ROWS / 2) * COLS + c[i] + COLS / 2
      return index
    }))
  }
  return positions.get(dataset)
}

/** One month of a layer's series, read live: values on the (row, column) array, NaN where missing.
 *  onProgress(share of row groups read, bytes so far, bytes expected) is called as row groups arrive. */
export async function loadSeries(layer, year, month, onProgress = () => {}) {
  const stats = { bytes: 0, requests: 0, network: 0, active: 0, t0: performance.now() }
  const progress = { total: 0, done: 0, expected: 0,
    report: () => onProgress(progress.done / Math.max(progress.total, 1), stats.bytes, progress.expected) }
  const url = `${SOURCE_COOP}/${layer.dataset}/series/year=${year}/${String(month).padStart(2, '0')}.parquet`
  stats.firstOfDataset = !positions.has(layer.dataset)
  const [index, cols] = await Promise.all([cellPositions(layer.dataset, stats, progress),
    readColumns(url, [[layer.column, Float32Array]], stats, progress)])
  stats.read = performance.now() - stats.t0
  const v = cols[layer.column]
  if (v.length !== index.length) throw new Error(`${url}: ${v.length} rows, cells.parquet has ${index.length}`)
  const values = new Float32Array(ROWS * COLS).fill(NaN), factor = layer.perMonth ? layer.perMonth(month) : 1
  for (let i = 0; i < v.length; i++) values[index[i]] = v[i] * factor
  stats.cells = v.length
  return { values, stats }
}

const newStats = () => ({ bytes: 0, requests: 0, network: 0, active: 0, t0: performance.now() })
const factor = (layer, month) => (layer.perMonth ? layer.perMonth(month) : 1)

/** Value of a row-group statistic (hyparquet gives min_value/max_value, older writers min/max). */
function bounds(group, name) {
  const st = group.columns.find(c => c.meta_data.path_in_schema.join('.') === name)?.meta_data.statistics
  const lo = st?.min_value ?? st?.min, hi = st?.max_value ?? st?.max
  return lo === undefined || hi === undefined ? [-Infinity, Infinity] : [Number(lo), Number(hi)]
}

/** The rows of one cell in a file whose rows run along the Hilbert curve: a row group covers a compact area, so
 *  its statistics on grid_row_u and grid_col_r show which row groups can hold the cell (usually one), and only
 *  those are read. Rows come back as arrays of `columns`, which must start with grid_row_u, grid_col_r. */
async function cellRows(url, row, col, columns, stats) {
  const file = await countedFile(url, stats)
  const metadata = await parquetMetadataAsync(file, { initialFetchSize: TAIL })
  const reads = [], found = []
  for (let start = 0, g = 0; g < metadata.row_groups.length; g++) {
    const group = metadata.row_groups[g], end = start + Number(group.num_rows)
    const [r0, r1] = bounds(group, 'grid_row_u'), [c0, c1] = bounds(group, 'grid_col_r')
    if (r0 <= row && row <= r1 && c0 <= col && col <= c1) reads.push([start, end])
    start = end
  }
  await Promise.all(reads.map(([rowStart, rowEnd]) => parquetRead({
    file, metadata, compressors, columns, rowStart, rowEnd,
    onComplete(rows) { for (const r of rows) if (Number(r[0]) === row && Number(r[1]) === col) found.push(r) },
  })))
  stats.rowGroups = reads.length
  return found
}

const bandOf = row => String(Math.floor(((row + ROWS / 2) * 20) / ROWS)).padStart(2, '0')      // as majortom.climate

/** One cell's monthly normals: {mean[12], std[12], stats}, from its climatology band file. */
export async function loadCellClimatology(layer, row, col) {
  const stats = newStats(), mean = new Array(12).fill(NaN), std = new Array(12).fill(NaN)
  const rows = await cellRows(`${SOURCE_COOP}/${layer.dataset}/climatology/band=${bandOf(row)}.parquet`, row, col,
    ['grid_row_u', 'grid_col_r', 'month', `${layer.column}_mean`, `${layer.column}_std`], stats)
  for (const [, , m, mu, sd] of rows) {
    const f = factor(layer, Number(m))
    mean[m - 1] = mu === null ? NaN : mu * f; std[m - 1] = sd === null ? NaN : sd * f
  }
  stats.read = performance.now() - stats.t0
  return { mean, std, stats }
}

/** One cell's every month from the dataset's by_cell layout: its band file, the row group holding its series. */
export async function loadByCellSeries(layer, row, col) {
  const stats = newStats(), [a, b] = layer.series, values = new Array((b - a + 1) * 12).fill(NaN)
  const rows = await cellRows(`${SOURCE_COOP}/${layer.dataset}/by_cell/band=${bandOf(row)}.parquet`, row, col,
    ['grid_row_u', 'grid_col_r', 'year', 'month', layer.column], stats)
  for (const [, , y, m, v] of rows) if (v !== null) values[(y - a) * 12 + m - 1] = v * factor(layer, Number(m))
  stats.read = performance.now() - stats.t0
  return { values, stats }
}

/** One month of the dataset's overview tables (4 x 4 block means, level 2): values on the block array. */
export async function loadOverviewMonth(layer, year, month, onProgress = () => {}) {
  const stats = newStats(), level = 2
  const url = `${SOURCE_COOP}/${layer.dataset}/overviews/year=${year}/${String(month).padStart(2, '0')}.parquet`
  const file = await countedFile(url, stats)
  const w = Math.ceil(COLS / 2 ** level), values = new Float32Array(Math.ceil(ROWS / 2 ** level) * w).fill(NaN)
  const f = factor(layer, month)
  await parquetRead({ file, compressors, initialFetchSize: TAIL, columns: ['block_row', 'block_col', layer.column],
    onComplete(rows) { for (const [r, c, v] of rows) if (v !== null) values[r * w + c] = v * f } })
  onProgress(1, stats.bytes, stats.bytes)
  stats.read = performance.now() - stats.t0; stats.level = level
  return { values, stats }
}

/** The row of a cell in every file of a dataset (the files share cells.parquet's order), or -1. */
export async function cellRow(dataset, row, col) {
  const quiet = { total: 0, done: 0, expected: 0, report() {} }
  const index = await cellPositions(dataset, newStats(), quiet)
  return index.indexOf((row + ROWS / 2) * COLS + col + COLS / 2)
}

/** Every month of one cell's series, read from the monthly files PARALLEL at a time: per file its tail (footer)
 *  and the column chunk of the row group holding the cell, the whole chunk, as Parquet stores it.
 *  onValue(k, value) as months arrive (k counts months from the series' first January); stop() ends early. */
export function loadCellSeries(layer, rowIndex, onValue) {
  const stats = newStats(), queue = []
  const [a, b] = layer.series
  for (let y = a, k = 0; y <= b; y++) for (let m = 1; m <= 12; m++, k++) queue.push([k, y, m])
  stats.months = queue.length
  let stopped = false, tailSize = TAIL         // after the first footer: just that size and a margin (files are alike)
  const done = Promise.all(Array.from({ length: PARALLEL }, async () => {
    for (let job; !stopped && (job = queue.shift());) {
      const [k, y, m] = job
      const url = `${SOURCE_COOP}/${layer.dataset}/series/year=${y}/${String(m).padStart(2, '0')}.parquet`
      let value = NaN
      try {
        const file = await countedFile(url, stats, tailSize)
        const metadata = await parquetMetadataAsync(file, { initialFetchSize: tailSize })
        tailSize = Math.min(TAIL, Math.max(tailSize === TAIL ? 0 : tailSize, metadata.metadata_length + 8 + 8192))
        await parquetRead({ file, metadata, compressors, columns: [layer.column], rowStart: rowIndex, rowEnd: rowIndex + 1,
          onComplete(rows) { const v = rows[0]?.[0]; if (v !== null && v !== undefined) value = v * factor(layer, m) } })
      } catch { /* a missing month stays NaN */ }
      onValue(k, value, stats)
    }
  })).then(() => { stats.read = performance.now() - stats.t0; return stats })
  return { done, stop: () => { stopped = true }, stats }
}

// ---- Zarr test store (docs/evidence/climate_zarr_test.py): the same values on the (row, column) array ---------

/** A zarrita store for the layer's test store whose requests are counted like the Parquet ones. A missing chunk
 *  (no values in it, so not written) answers 404, which zarrita reads as all NaN. */
function zarrStore(layer, stats, onResponse = () => {}) {
  return new zarr.FetchStore(`${SOURCE_COOP}/${layer.zarr.path}`, {
    async fetch(request) {
      if (stats.active++ === 0) stats.since = performance.now()
      try {
        const r = await fetch(request), body = await r.arrayBuffer()
        stats.requests += 1; stats.bytes += body.byteLength; onResponse()
        return new Response(body, { status: r.status, headers: r.headers })
      } finally {
        if (--stats.active === 0) stats.network += performance.now() - stats.since
      }
    },
  })
}

/** One month (k-th of the store's map months) at `level` (0: cells, 1: 2 x 2, 2: 4 x 4 block means), whole. */
export async function loadZarrMonth(layer, k, level, onProgress = () => {}) {
  const stats = newStats(), chunks = { done: 0, total: 1 }
  const store = zarrStore(layer, stats, () => { chunks.done++; onProgress(chunks.done / chunks.total, stats.bytes, 0) })
  const array = await zarr.open(zarr.root(store).resolve(`map/${level}`), { kind: 'array' })
  chunks.total = Math.ceil(array.shape[1] / array.chunks[1]) * Math.ceil(array.shape[2] / array.chunks[2]); chunks.done = 0
  const { data } = await zarr.get(array, [k, null, null])
  stats.read = performance.now() - stats.t0; stats.level = level; stats.chunks = chunks.total
  return { values: data, stats }
}

/** One cell's every month from the store's series array: one chunk (all months of 32 x 32 cells) in its shard. */
export async function loadZarrCellSeries(layer, row, col) {
  const stats = newStats(), store = zarrStore(layer, stats)
  const array = await zarr.open(zarr.root(store).resolve('series'), { kind: 'array' })
  const { data } = await zarr.get(array, [null, row + ROWS / 2, col + COLS / 2])
  stats.read = performance.now() - stats.t0
  return { values: Array.from(data), stats }
}

/** One month of a layer's climatology preview image: RGBA pixels, width x width/2. */
export async function loadPreview(layer, month) {
  const stats = { bytes: 0, requests: 1, t0: performance.now() }
  // revalidated with the server: a cached image from an earlier build (another layout) is never used
  const blob = await (await fetch(layer.preview.replace('{MM}', String(month).padStart(2, '0')) + location.search, { cache: 'no-cache' })).blob()
  stats.bytes = blob.size
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })  // colours as stored
  if (bitmap.width !== COLS || bitmap.height !== ROWS) {
    throw new Error(`preview image is ${bitmap.width} x ${bitmap.height}, expected ${COLS} x ${ROWS} (one pixel per cell); rebuild it with build_images.py`)
  }
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height), ctx = canvas.getContext('2d')
  ctx.drawImage(bitmap, 0, 0)
  stats.read = performance.now() - stats.t0
  return { rgba: ctx.getImageData(0, 0, bitmap.width, bitmap.height).data, width: bitmap.width, stats }
}
