// AlphaEarth embeddings (tge-labs/aef on source.coop): a cell's window of its yearly COG tile, shown as a PCA of all
// 64 bands, or one band.
//
//   footprint row (tile_row, tile_rows) -> footprints/AEF-tiles.parquet, that tile's rows (page index) -> the year's COG
//   -> geotiff.js over range requests (a counting client: bytes for the loading flag; zstd decoder in zstd.js)
//      -> the level's image (0: 10 m, 1: 20 m, 2: 40 m) -> the cell's 10,680 m window in the tile's UTM zone, int8
//         (the COG is band-interleaved: only the tiles under the window, of the bands asked for, are read)
//   -> PCA: mean and 64 x 64 covariance from up to 50,000 pixels, the three leading components (power iteration
//      with deflation), every pixel projected, each component stretched 2-98% -> RGB PNG; one band: grey, 2-98%
//   The COGs are stored south-up (row 0 at the tile's southern edge): the origin and the row direction come from the
//   file's own transform, and rows are flipped so north is up.
//   Pixels outside the tile (cells at a tile edge) and no-data (-128) are transparent. PCA colours are the cell's
//   own: the same colour means similar embeddings within a cell, not across cells.

import { fromCustomClient } from 'https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm'
import proj4 from 'https://cdn.jsdelivr.net/npm/proj4@2.12.1/+esm'
import { compressors, newStats, openFile, parquetRead } from './parquet.js'
import './zstd.js'

const HALF = 5340, NODATA = -128, FIT = 50_000

export const LEVELS = [[0, '10 m'], [1, '20 m'], [2, '40 m']]

/** The views offered: PCA at 40 m (the default, about 38 MB a cell) and 10 m (about 117 MB), then single bands. */
export function aefViews() {
  return [{ id: 'pca:2', label: 'PCA of 64 bands, 40 m' }, { id: 'pca:0', label: 'PCA of 64 bands, 10 m (~120 MB)' },
    ...Array.from({ length: 64 }, (_, b) => ({ id: `band:${b}`, label: `Band A${String(b).padStart(2, '0')}, 10 m` }))]
}

/** A geotiff.js client that counts what it reads into `stats`. */
function countingClient(url, stats) {
  return {
    async request({ headers, signal } = {}) {
      const r = await fetch(url, { headers, signal }), data = await r.arrayBuffer()
      stats.bytes += data.byteLength; stats.requests += 1
      return { ok: r.ok, status: r.status, getHeader: name => r.headers.get(name), getData: async () => data }
    },
  }
}

/** The tile rows of a cell (one per year): {year, epsg, west, north, path}. */
async function tileRows(info, sample, stats) {
  const url = new URL(`footprints/${info.tiles}`, location.href).href.split('?')[0] + location.search
  const { file, metadata } = await openFile(url, stats)
  let rows = []
  await parquetRead({ file, metadata, compressors, useOffsetIndex: true, rowStart: sample.tile_row,
    rowEnd: sample.tile_row + sample.tile_rows, rowFormat: 'object', onComplete(r) { rows = r } })
  return rows
}

/** One view of the cell's window in the sample's year: {src, size, values (one band: for the cursor), corners, stats}. */
export async function renderAef(info, sample, viewId, stats = newStats()) {
  const [kind, arg] = viewId.split(':'), level = kind === 'pca' ? Number(arg) : 0
  const tile = (await tileRows(info, sample, stats)).find(r => Number(r.year) === sample.year)
  if (!tile) throw new Error(`no AlphaEarth tile for ${sample.year}`)
  const epsg = Number(tile.epsg)
  const utm = proj4(`+proj=utm +zone=${epsg % 100}${epsg > 32700 ? ' +south' : ''} +datum=WGS84 +units=m +no_defs`)
  const [cx, cy] = utm.forward([sample.centre_lon, sample.centre_lat])
  const tiff = await fromCustomClient(countingClient(`${info.files}${tile.path}`, stats))
  const [, , , left, , dy, , top] = (await tiff.getImage(0)).fileDirectory.ModelTransformation   // dy > 0: south-up
  const image = await tiff.getImage(level), k = 2 ** level, size = Math.round(2 * HALF / (10 * k))
  const x0 = Math.round((cx - HALF - left) / (10 * k))
  const y0 = Math.round(Math.min((cy + HALF - top) / (dy * k), (cy - HALF - top) / (dy * k)))
  const W = image.getWidth(), H = image.getHeight()
  const win = [Math.max(x0, 0), Math.max(y0, 0), Math.min(x0 + size, W), Math.min(y0 + size, H)]
  const samples = kind === 'pca' ? Array.from({ length: image.getSamplesPerPixel() }, (_, b) => b) : [Number(arg)]
  stats.expected = stats.bytes + expectedBytes(image, win, samples)
  const read = await image.readRasters({ window: win, samples, interleave: false })
  stats.phase = kind === 'pca' ? 'PCA' : 'Drawing'
  const bands = read.map(part => place(part, win, x0, y0, size, dy > 0))   // full window, north up, NODATA outside
  const rgba = kind === 'pca' ? pcaRgba(bands, size) : greyRgba(bands[0], size)
  const canvas = new OffscreenCanvas(size, size)
  canvas.getContext('2d').putImageData(new ImageData(rgba, size, size), 0, 0)
  const src = URL.createObjectURL(await canvas.convertToBlob({ type: 'image/png' }))
  const corners = [[cx - HALF, cy + HALF], [cx + HALF, cy + HALF], [cx + HALF, cy - HALF], [cx - HALF, cy - HALF]]
    .map(p => utm.inverse(p))
  stats.read = performance.now() - stats.t0
  const values = kind === 'band' ? { width: size, values: Float32Array.from(bands[0], v => (v === NODATA ? NaN : v)), unit: '' } : null
  return { src, size, values, corners, stats }
}

/** Compressed bytes of the tiles under the window, for the bands read (planar COG: tiles band by band). */
function expectedBytes(image, [a, b, c, d], samples) {
  const fd = image.fileDirectory, T = fd.TileWidth, across = Math.ceil(image.getWidth() / T)
  const per = across * Math.ceil(image.getHeight() / fd.TileLength)
  let sum = 0
  for (const s of samples) {
    for (let r = Math.floor(b / T); r <= Math.floor((d - 1) / T); r++) {
      for (let q = Math.floor(a / T); q <= Math.floor((c - 1) / T); q++) sum += fd.TileByteCounts[s * per + r * across + q]
    }
  }
  return sum
}

function place(part, [a, b, c, d], x0, y0, size, flip) {
  const out = new Int8Array(size * size).fill(NODATA), w = c - a
  for (let y = b; y < d; y++) {
    const row = flip ? size - 1 - (y - y0) : y - y0
    out.set(part.subarray((y - b) * w, (y - b + 1) * w), row * size + (a - x0))
  }
  return out
}

/** 2nd and 98th percentile of a sample of values. */
function range(values) {
  const v = Float64Array.from(values).sort()
  if (!v.length) return [0, 1]
  const lo = v[Math.floor(v.length * 0.02)], hi = v[Math.floor(v.length * 0.98)]
  return hi > lo ? [lo, hi] : [lo, lo + 1]
}

function greyRgba(band, size) {
  const out = new Uint8ClampedArray(size * size * 4), picked = []
  for (let i = 0; i < band.length; i += Math.max(1, Math.floor(band.length / FIT))) if (band[i] !== NODATA) picked.push(band[i])
  const [lo, hi] = range(picked)
  for (let i = 0; i < band.length; i++) {
    if (band[i] === NODATA) continue
    const g = 255 * Math.min(Math.max((band[i] - lo) / (hi - lo), 0), 1)
    out[i * 4] = g; out[i * 4 + 1] = g; out[i * 4 + 2] = g; out[i * 4 + 3] = 255
  }
  return out
}

/** The three leading principal components of the 64 bands, as RGB; no-data transparent. */
function pcaRgba(bands, size) {
  const n = size * size, B = bands.length, valid = []
  for (let i = 0; i < n; i++) if (bands[0][i] !== NODATA) valid.push(i)
  const out = new Uint8ClampedArray(n * 4)
  if (valid.length < 2) return out
  const step = Math.max(1, Math.floor(valid.length / FIT)), fit = valid.filter((_, k) => k % step === 0)
  const mean = new Float64Array(B), cov = new Float64Array(B * B), x = new Float64Array(B)
  for (const i of fit) for (let b = 0; b < B; b++) mean[b] += bands[b][i] / fit.length
  for (const i of fit) {
    for (let b = 0; b < B; b++) x[b] = bands[b][i] - mean[b]
    for (let b = 0; b < B; b++) for (let c = b; c < B; c++) cov[b * B + c] += x[b] * x[c]
  }
  for (let b = 0; b < B; b++) for (let c = 0; c < b; c++) cov[b * B + c] = cov[c * B + b]
  const comps = leading(cov, B, 3)
  const proj = comps.map(() => new Float32Array(n))
  for (const i of valid) {
    for (let k = 0; k < 3; k++) {
      let s = 0
      for (let b = 0; b < B; b++) s += (bands[b][i] - mean[b]) * comps[k][b]
      proj[k][i] = s
    }
  }
  const ranges = proj.map(p => range(fit.map(i => p[i])))
  for (const i of valid) {
    for (let k = 0; k < 3; k++) out[i * 4 + k] = 255 * Math.min(Math.max((proj[k][i] - ranges[k][0]) / (ranges[k][1] - ranges[k][0]), 0), 1)
    out[i * 4 + 3] = 255
  }
  return out
}

/** The `count` leading eigenvectors of a symmetric B x B matrix: power iteration, deflating after each. Signs are
 *  fixed (largest loading positive) so the same data gives the same colours. */
function leading(cov, B, count) {
  const m = Float64Array.from(cov), out = []
  for (let k = 0; k < count; k++) {
    let v = Float64Array.from({ length: B }, (_, b) => 1 + b / B), lambda = 0
    for (let it = 0; it < 200; it++) {
      const w = new Float64Array(B)
      for (let b = 0; b < B; b++) { let s = 0; for (let c = 0; c < B; c++) s += m[b * B + c] * v[c]; w[b] = s }
      const norm = Math.hypot(...w)
      if (!norm) break
      lambda = norm; v = w.map(e => e / norm)
    }
    let big = 0
    for (let b = 1; b < B; b++) if (Math.abs(v[b]) > Math.abs(v[big])) big = b
    if (v[big] < 0) v = v.map(e => -e)
    for (let b = 0; b < B; b++) for (let c = 0; c < B; c++) m[b * B + c] -= lambda * v[b] * v[c]
    out.push(v)
  }
  return out
}
