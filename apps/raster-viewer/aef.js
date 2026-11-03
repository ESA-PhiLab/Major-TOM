// AlphaEarth embeddings (tge-labs/aef on source.coop): a cell's window of its yearly COG tiles, shown as a PCA of the
// 64 bands (fast: a coarse PCA first, then the high-res image from a few bands), or one band.
//
//   footprint row (tile_row, tile_rows: lists, one entry per tile under the cell's 10,680 m window, its own first)
//     -> footprints/AEF-tiles.parquet, each tile's rows (page index) -> the year's COGs
//   each tile gives the part of the window it covers; the parts are joined (a zone's tiles share one 81,920 m grid,
//   a whole number of pixels at every level, so they fit exactly); 24.6% of cells need more than one tile
//   PCA, in two steps (each source.coop request waits ~0.7 s and few run at once, so requests set the time):
//     1. preview: each tile's whole 160 m level, all 64 bands in one request (~10 MB; the level's tiles are stored
//        together) -> the window (67 px) -> PCA (three leading components) -> shown at once (onPreview)
//        the level is kept: other cells of the same tile and year skip this read
//     2. high-res: the 8 bands that best predict the three components (forward selection, least squares on the
//        160 m pixels; R² 0.95-0.98 against the full 64-band PCA, measured at 40 m for three places) -> those bands at
//        10 m (geotiff.js, only the tiles under the window) -> the fitted mapping -> 1,068 px
//   geotiff.js reads in 1 MB blocks: the COG's headers (one per level) in one request, a band's adjacent tiles in one
//   one band: that band at 10 m, grey, 2-98% stretch; downloads: the joined window's bands at 10 m (download.js)
//   The COGs are stored south-up (row 0 at the tile's southern edge): the origin and the row direction come from
//   each file's own transform, and rows are flipped so north is up.
//   Pixels no tile covers (a zone edge, the coast) and no-data (-128) are transparent. PCA colours are the cell's
//   own: the same colour means similar embeddings within a cell, not across cells.

import { fromCustomClient, getDecoder } from 'https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm'
import proj4 from 'https://cdn.jsdelivr.net/npm/proj4@2.12.1/+esm'
import { compressors, newStats, openFile, parquetRead } from './parquet.js'
import './zstd.js'

const HALF = 5340, NODATA = -128, FIT = 50_000, OVERVIEW = 4, HIGH = 0, CHOSEN = 8    // levels: 4 = 160 m, 0 = 10 m

/** The views offered: the PCA (default), then single bands. */
export function aefViews() {
  return [{ id: 'pca', label: 'PCA proxy of 64 bands, 10 m' },            // the 160 m PCA, predicted from 8 bands
    ...Array.from({ length: 64 }, (_, b) => ({ id: `band:${b}`, label: `Band A${String(b).padStart(2, '0')}, 10 m` }))]
}

/** A geotiff.js client that counts what it reads into `client.stats`: the read under way, set again by openCog. */
function countingClient(url, stats) {
  const client = {
    stats,
    async request({ headers, signal } = {}) {
      const r = await fetch(url, { headers, signal }), data = await r.arrayBuffer()
      client.stats.bytes += data.byteLength; client.stats.requests += 1
      return { ok: r.ok, status: r.status, getHeader: name => r.headers.get(name), getData: async () => data }
    },
  }
  return client
}

/** One tile's rows (one per year): {year, epsg, west, north, path}. */
async function tileRows(info, start, count, stats) {
  const url = new URL(`footprints/${info.tiles}`, location.href).href.split('?')[0] + location.search
  const { file, metadata } = await openFile(url, stats)
  let rows = []
  await parquetRead({ file, metadata, compressors, useOffsetIndex: true, rowStart: start, rowEnd: start + count,
    rowFormat: 'object', onComplete(r) { rows = r } })
  return rows
}

const listOf = v => (Array.isArray(v) || ArrayBuffer.isView(v) ? [...v].map(Number) : [Number(v)])

/** The cell in its UTM zone, and its tiles' files in the sample's year: {epsg, cx, cy, corners, urls}. */
export async function aefCell(info, sample, stats = newStats()) {
  const starts = listOf(sample.tile_row), counts = listOf(sample.tile_rows)
  const rows = await Promise.all(starts.map((s, k) => tileRows(info, s, counts[k], stats)))
  const tiles = rows.map(r => r.find(x => Number(x.year) === sample.year)).filter(Boolean)
  if (!tiles.length) throw new Error(`no AlphaEarth tile for ${sample.year}`)
  const epsg = Number(tiles[0].epsg)
  const utm = proj4(`+proj=utm +zone=${epsg % 100}${epsg > 32700 ? ' +south' : ''} +datum=WGS84 +units=m +no_defs`)
  const [cx, cy] = utm.forward([sample.centre_lon, sample.centre_lat])
  const corners = [[cx - HALF, cy + HALF], [cx + HALF, cy + HALF], [cx + HALF, cy - HALF], [cx - HALF, cy - HALF]]
    .map(q => utm.inverse(q))
  return { epsg, cx, cy, corners, urls: tiles.map(x => `${info.files}${x.path}`) }
}

const opened = new Map()                       // COG url -> {tiff, client, left, top, dy}, most recently used last
const KEEP = 4                                 // COGs kept open; older ones are dropped with their block cache

/** A COG, opened once (its headers in one 1 MB block), with its origin and row direction; its reads are counted
 *  into `stats` from now on. At most KEEP stay open, each caching at most 16 blocks (16 MB). */
async function openCog(url, stats) {
  if (opened.has(url)) { const c = opened.get(url); opened.delete(url); opened.set(url, c) }
  else {
    opened.set(url, (async () => {
      const client = countingClient(url, stats)
      const tiff = await fromCustomClient(client, { blockSize: 2 ** 20, cacheSize: 16 })
      const [, , , left, , dy, , top] = (await tiff.getImage(0)).fileDirectory.ModelTransformation   // dy > 0: south-up
      return { url, tiff, client, left, top, dy }
    })().catch(e => { opened.delete(url); throw e }))
    while (opened.size > KEEP) opened.delete(opened.keys().next().value)
  }
  const cog = await opened.get(url)
  cog.client.stats = stats
  return cog
}

/** Every tile's part of the window, joined: one Int8Array per band (NODATA where no tile covers). */
function join(parts, size) {
  return parts[0].map((first, b) => {
    const out = Int8Array.from(first)
    for (const part of parts.slice(1)) {
      const band = part[b]
      for (let i = 0; i < out.length; i++) if (out[i] === NODATA) out[i] = band[i]
    }
    return out
  })
}

/** Bands of the cell's window at one level, read from every tile and joined: {bands, size, west, north, res}. */
export async function aefWindow(info, sample, level, samples, stats = newStats()) {
  const cell = await aefCell(info, sample, stats), cogs = await Promise.all(cell.urls.map(u => openCog(u, stats)))
  const k = 2 ** level, size = Math.round(2 * HALF / (10 * k)), res = 10 * k
  const parts = await Promise.all(cogs.map(async c => {
    const image = await c.tiff.getImage(level), w = windowAt(c.left, c.top, c.dy, level, cell.cx, cell.cy)
    return readWindow(image, w, samples, stats, c.dy > 0)
  }))
  const w = windowAt(cogs[0].left, cogs[0].top, cogs[0].dy, level, cell.cx, cell.cy)
  return { bands: join(parts.filter(Boolean), size), size, res, west: w.west, north: w.north, cell }
}

/** One view of the cell's window in the sample's year: {src, size, values (one band: for the cursor), corners, stats}.
 *  For the PCA, onPreview(view) gets the 160 m image as soon as it is drawn. */
export async function renderAef(info, sample, viewId, stats = newStats(), onPreview = () => {}) {
  const cell = await aefCell(info, sample, stats)
  const done = async (rgba, size, values = null) => {
    const src = await png(rgba, size)
    stats.read = performance.now() - stats.t0
    return { src, size, values, corners: cell.corners, stats }
  }

  if (viewId.startsWith('band:')) {
    const window = await aefWindow(info, sample, 0, [Number(viewId.split(':')[1])], stats), { bands: [band], size } = window
    const values = { width: size, values: Float32Array.from(band, v => (v === NODATA ? NaN : v)), unit: '' }
    return { ...(await done(greyRgba(band, size), size, values)), window }            // window: for the band's GeoTIFF
  }

  stats.phase = 'Overview, 160 m'
  const cogs = await Promise.all(cell.urls.map(u => openCog(u, stats)))
  const size8 = Math.round(2 * HALF / (10 * 2 ** OVERVIEW))
  const coarse = join(await Promise.all(cogs.map(async c => {
    const overview = await overviewOf(c, stats), w = windowAt(c.left, c.top, c.dy, OVERVIEW, cell.cx, cell.cy)
    return overview.bands.map(band => cut(band, overview.width, overview.height, w, c.dy > 0))
  })), size8)
  const pca = fitPca(coarse, size8)
  if (!pca) return done(new Uint8ClampedArray(size8 * size8 * 4), size8)
  onPreview({ ...(await done(rgbOf(pca.scores, pca.valid, size8), size8)), label: 'PCA of 64 bands, 160 m' })

  stats.phase = `Loading high-res: ${CHOSEN} bands at 10 m`
  const { chosen, weights } = selectBands(coarse, pca, CHOSEN)
  const { bands: fine, size } = await aefWindow(info, sample, HIGH, chosen, stats)
  const n = size * size, valid = [], scores = [0, 1, 2].map(() => new Float32Array(n))
  for (let i = 0; i < n; i++) {
    if (fine[0][i] === NODATA) continue
    valid.push(i)
    for (let k = 0; k < 3; k++) {
      let s = weights[chosen.length][k]                                        // intercept
      for (let j = 0; j < chosen.length; j++) s += fine[j][i] * weights[j][k]
      scores[k][i] = s
    }
  }
  return done(rgbOf(scores, valid, size), size)
}

/** The cell's window on one level: its size, its position in that level's pixels (rows counted as stored), and its
 *  west and north edges in metres. */
function windowAt(left, top, dy, level, cx, cy) {
  const k = 2 ** level, res = 10 * k, size = Math.round(2 * HALF / res)
  const x0 = Math.round((cx - HALF - left) / res)
  const y0 = Math.round(Math.min((cy + HALF - top) / (dy * k), (cy - HALF - top) / (dy * k)))
  const yEdges = [top + y0 * dy * k, top + (y0 + size) * dy * k]
  return { level, size, x0, y0, west: left + x0 * res, north: Math.max(...yEdges) }
}

const overviews = new Map()                    // COG url -> its 160 m level, all bands (six kept: ~16 MB each)

/** The preview level of a COG (160 m), every band, from one request over the level's tiles (stored together), kept. */
function overviewOf({ url, tiff }, stats) {
  if (!overviews.has(url)) {
    overviews.set(url, (async () => {
      const image = await tiff.getImage(OVERVIEW), fd = image.fileDirectory, offsets = fd.TileOffsets, counts = fd.TileByteCounts
      const start = Math.min(...offsets), end = Math.max(...offsets.map((o, i) => o + counts[i]))
      stats.expected = Math.max(stats.expected ?? 0, stats.bytes) + (end - start)
      const r = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } })
      if (r.status !== 206) throw new Error(`AlphaEarth overview: HTTP ${r.status}`)
      const data = await r.arrayBuffer()
      stats.bytes += data.byteLength; stats.requests += 1
      const decoder = await getDecoder(fd), W = image.getWidth(), H = image.getHeight(), T = fd.TileWidth
      const across = Math.ceil(W / T), down = Math.ceil(H / fd.TileLength), per = across * down
      const bands = []
      for (let b = 0; b < image.getSamplesPerPixel(); b++) {
        const band = new Int8Array(W * H)
        for (let r = 0; r < down; r++) {
          for (let q = 0; q < across; q++) {
            const t = b * per + r * across + q
            const raw = new Int8Array(await decoder.decode(fd, data.slice(offsets[t] - start, offsets[t] - start + counts[t])))
            for (let y = 0; y < T && r * T + y < H; y++) {
              const w = Math.min(T, W - q * T)
              band.set(raw.subarray(y * T, y * T + w), (r * T + y) * W + q * T)
            }
          }
        }
        bands.push(band)
      }
      if (overviews.size > 6) overviews.delete(overviews.keys().next().value)
      return { width: W, height: H, bands }
    })().catch(e => { overviews.delete(url); throw e }))
  }
  return overviews.get(url)
}

/** The window's pixels of some bands in one COG, read through geotiff.js (only the tiles under it), north up;
 *  null when the COG holds none of the window. */
async function readWindow(image, w, samples, stats, flip) {
  const W = image.getWidth(), H = image.getHeight()
  const box = [Math.max(w.x0, 0), Math.max(w.y0, 0), Math.min(w.x0 + w.size, W), Math.min(w.y0 + w.size, H)]
  if (box[2] <= box[0] || box[3] <= box[1]) return null                 // this tile holds none of the window
  stats.expected = Math.max(stats.expected ?? 0, stats.bytes) + expectedBytes(image, box, samples)
  const parts = await image.readRasters({ window: box, samples, interleave: false })
  return parts.map(part => place(part, box, w, flip))
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

/** A read box (part of the window inside the image) placed into the full window, north up, NODATA elsewhere. */
function place(part, [a, b, c, d], w, flip) {
  const out = new Int8Array(w.size * w.size).fill(NODATA), width = c - a
  for (let y = b; y < d; y++) {
    const row = flip ? w.size - 1 - (y - w.y0) : y - w.y0
    out.set(part.subarray((y - b) * width, (y - b + 1) * width), row * w.size + (a - w.x0))
  }
  return out
}

/** The window cut from a whole level held in memory. */
function cut(band, W, H, w, flip) {
  const box = [Math.max(w.x0, 0), Math.max(w.y0, 0), Math.min(w.x0 + w.size, W), Math.min(w.y0 + w.size, H)]
  const width = box[2] - box[0], part = new Int8Array(Math.max(width, 0) * Math.max(box[3] - box[1], 0))
  for (let y = box[1]; y < box[3]; y++) part.set(band.subarray(y * W + box[0], y * W + box[2]), (y - box[1]) * width)
  return place(part, box, w, flip)
}

/** PCA of the window's bands: {mean, components (3 x 64), scores (3 arrays), valid (pixel indices), fit (indices)}. */
function fitPca(bands, size) {
  const n = size * size, B = bands.length, valid = []
  for (let i = 0; i < n; i++) if (bands[0][i] !== NODATA) valid.push(i)
  if (valid.length < 2) return null
  const step = Math.max(1, Math.floor(valid.length / FIT)), fit = valid.filter((_, k) => k % step === 0)
  const mean = new Float64Array(B), cov = new Float64Array(B * B), x = new Float64Array(B)
  for (const i of fit) for (let b = 0; b < B; b++) mean[b] += bands[b][i] / fit.length
  for (const i of fit) {
    for (let b = 0; b < B; b++) x[b] = bands[b][i] - mean[b]
    for (let b = 0; b < B; b++) for (let c = b; c < B; c++) cov[b * B + c] += x[b] * x[c]
  }
  for (let b = 0; b < B; b++) for (let c = 0; c < b; c++) cov[b * B + c] = cov[c * B + b]
  const components = leading(cov, B, 3), scores = components.map(() => new Float32Array(n))
  for (const i of valid) {
    for (let k = 0; k < 3; k++) {
      let s = 0
      for (let b = 0; b < B; b++) s += (bands[b][i] - mean[b]) * components[k][b]
      scores[k][i] = s
    }
  }
  return { mean, components, scores, valid, fit }
}

/** The `count` bands that best predict the three components (forward selection on R², least squares with an
 *  intercept, from the normal equations), and the weights: rows per chosen band, then the intercept; 3 columns. */
function selectBands(bands, pca, count) {
  const B = bands.length, I = B, pixels = pca.fit, Y = pca.scores
  const G = new Float64Array((B + 1) * (B + 1)), XY = Array.from({ length: B + 1 }, () => new Float64Array(3))
  const value = (b, i) => (b === I ? 1 : bands[b][i])
  for (const i of pixels) {
    for (let b = 0; b <= B; b++) {
      const vb = value(b, i)
      for (let c = b; c <= B; c++) G[b * (B + 1) + c] += vb * value(c, i)
      for (let k = 0; k < 3; k++) XY[b][k] += vb * Y[k][i]
    }
  }
  for (let b = 0; b <= B; b++) for (let c = 0; c < b; c++) G[b * (B + 1) + c] = G[c * (B + 1) + b]
  const explained = set => {                         // sum over components of the fitted values' sum of squares
    const w = solve(set.map(b => set.map(c => G[b * (B + 1) + c])), set.map(b => [...XY[b]]))
    if (!w) return [-Infinity, null]
    let s = 0
    for (let j = 0; j < set.length; j++) for (let k = 0; k < 3; k++) s += w[j][k] * XY[set[j]][k]
    return [s, w]
  }
  const chosen = []
  for (let step = 0; step < count; step++) {
    let best = -1, bestScore = -Infinity
    for (let b = 0; b < B; b++) {
      if (chosen.includes(b)) continue
      const [s] = explained([...chosen, b, I])
      if (s > bestScore) { bestScore = s; best = b }
    }
    chosen.push(best)
  }
  return { chosen, weights: explained([...chosen, I])[1] }
}

/** Solves A W = R (A: m x m, R: m x 3) by Gaussian elimination with partial pivoting; null if singular. */
function solve(A, R) {
  const m = A.length, a = A.map((row, i) => [...row, ...R[i]])
  for (let c = 0; c < m; c++) {
    let p = c
    for (let r = c + 1; r < m; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r
    if (Math.abs(a[p][c]) < 1e-9) return null
    ;[a[c], a[p]] = [a[p], a[c]]
    for (let r = 0; r < m; r++) {
      if (r === c) continue
      const f = a[r][c] / a[c][c]
      for (let k = c; k < m + 3; k++) a[r][k] -= f * a[c][k]
    }
  }
  return a.map((row, i) => row.slice(m).map(v => v / a[i][i]))
}

/** 2nd and 98th percentile of a sample of values. */
function range(values) {
  const v = Float64Array.from(values).sort()
  if (!v.length) return [0, 1]
  const lo = v[Math.floor(v.length * 0.02)], hi = v[Math.floor(v.length * 0.98)]
  return hi > lo ? [lo, hi] : [lo, lo + 1]
}

/** Three component images as RGB, each stretched 2-98% over a sample of the valid pixels; the rest transparent. */
function rgbOf(scores, valid, size) {
  const out = new Uint8ClampedArray(size * size * 4), step = Math.max(1, Math.floor(valid.length / FIT))
  const ranges = scores.map(s => range(valid.filter((_, k) => k % step === 0).map(i => s[i])))
  for (const i of valid) {
    for (let k = 0; k < 3; k++) out[i * 4 + k] = 255 * Math.min(Math.max((scores[k][i] - ranges[k][0]) / (ranges[k][1] - ranges[k][0]), 0), 1)
    out[i * 4 + 3] = 255
  }
  return out
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

async function png(rgba, size) {
  const canvas = new OffscreenCanvas(size, size)
  canvas.getContext('2d').putImageData(new ImageData(rgba, size, size), 0, 0)
  return URL.createObjectURL(await canvas.convertToBlob({ type: 'image/png' }))
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
