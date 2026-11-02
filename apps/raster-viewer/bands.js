// A sample's bands: read from its image file on Hugging Face, decoded and drawn in the browser.
//
//   image file row group (one per sample) -> band column = the original GeoTIFF bytes (LZW; kept for download)
//     -> geotiff.js -> values (S2: uint16 at 10, 20 or 60 m; S1: float32 linear backscatter at 7.5 m; DEM: float32 m)
//     -> a view: one band in grey (2-98% stretch), or a composite of three (S2: one stretch for the three) -> PNG
//   Every band of a sample covers the same 10,680 m square, so bands of different resolutions are combined by
//   sampling each at the composite's pixel positions (nearest).

import { fromArrayBuffer } from 'https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm'
import { compressors, newStats, openFile, parquetRead } from './parquet.js'
import { imageUrl } from './samples.js'

const NOT_BANDS = new Set(['product_id', 'grid_cell', 'product_datetime', 'thumbnail', 'compressed'])

/** The views offered for a dataset: composites first, then every band of its image files. */
export function views(name, bands) {
  const composites = {
    S2L2A: [['True colour (B04, B03, B02)', ['B04', 'B03', 'B02']], ['False colour (B08, B04, B03)', ['B08', 'B04', 'B03']],
      ['SWIR (B12, B8A, B04)', ['B12', 'B8A', 'B04']]],
    S1RTC: [['VV, VH, VV/VH (dB)', ['vv', 'vh', 'ratio']]],
    DEM: [['Elevation (colour)', ['DEM']]],
  }
  composites.S2L1C = composites.S2L2A
  return [...(composites[name] ?? []).map(([label, use]) => ({ id: `rgb:${use.join(',')}`, label, use })),
    ...bands.map(b => ({ id: b, label: b, use: [b] }))]
}

const raw = new Map()                      // "<url>#<row group>/<band>" -> GeoTIFF bytes

/** The band columns of a sample's image file (every column holding a raster). */
export async function bandNames(info, sample) {
  const { metadata } = await openFile(imageUrl(info, sample.part), newStats())
  return metadata.schema.slice(1).map(c => c.name).filter(n => !NOT_BANDS.has(n))
}

/** The GeoTIFF bytes of `bands` of one sample (read once, then kept): {name: Uint8Array}, and the read's stats. */
export async function readBands(info, sample, bands) {
  const stats = newStats(), url = imageUrl(info, sample.part), key = b => `${url}#${sample.parquet_row}/${b}`
  const missing = bands.filter(b => !raw.has(key(b)))
  if (missing.length) {
    const { file, metadata } = await openFile(url, stats)
    let rowStart = 0
    for (let g = 0; g < sample.parquet_row; g++) rowStart += Number(metadata.row_groups[g].num_rows)
    await parquetRead({ file, metadata, compressors, columns: missing, rowStart, rowEnd: rowStart + 1, rowFormat: 'object',
      utf8: false, onComplete(rows) { for (const b of missing) raw.set(key(b), rows[0][b]) } })
  }
  stats.read = performance.now() - stats.t0
  return [Object.fromEntries(bands.map(b => [b, raw.get(key(b))])), stats]
}

/** Values of one GeoTIFF: {width, height, values (typed array), noData}. */
async function decode(bytes) {
  const tiff = await fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
  const image = await tiff.getImage(), [values] = await image.readRasters()
  return { width: image.getWidth(), height: image.getHeight(), values, noData: image.getGDALNoData() }
}

/** 2nd and 98th percentile of the finite values (and not noData), from a sample of 100,000 pixels. */
function stretch(values, noData) {
  const step = Math.max(1, Math.floor(values.length / 100000)), picked = []
  for (let i = 0; i < values.length; i += step) {
    const v = values[i]
    if (Number.isFinite(v) && v !== noData) picked.push(v)
  }
  if (!picked.length) return [0, 1]
  picked.sort((a, b) => a - b)
  const lo = picked[Math.floor(picked.length * 0.02)], hi = picked[Math.floor(picked.length * 0.98)]
  return hi > lo ? [lo, hi] : [lo, lo + 1]
}

const dB = v => (v > 0 ? 10 * Math.log10(v) : NaN)
const TERRAIN = [[0, [26, 102, 51]], [0.3, [127, 176, 82]], [0.55, [222, 207, 140]], [0.8, [153, 102, 51]], [1, [250, 250, 250]]]

function terrain(t) {
  for (let k = 1; k < TERRAIN.length; k++) {
    const [t1, c1] = TERRAIN[k], [t0, c0] = TERRAIN[k - 1]
    if (t <= t1) { const f = (t - t0) / (t1 - t0); return c0.map((c, i) => c + f * (c1[i] - c)) }
  }
  return TERRAIN.at(-1)[1]
}

/** One view of a sample as a PNG object URL: {src, size, values (single band: for the cursor), stats}. */
export async function render(info, sample, view) {
  const needed = [...new Set(view.use.flatMap(b => (b === 'ratio' ? ['vv', 'vh'] : [b])))]
  const [bytes, stats] = await readBands(info, sample, needed)
  const bands = Object.fromEntries(await Promise.all(needed.map(async b => [b, await decode(bytes[b])])))
  const size = Math.max(...needed.map(b => bands[b].width))
  const s1 = 'vv' in bands, at = (band, x, y) => {             // value at composite pixel (x, y), nearest
    const k = band.width / size
    return band.values[Math.floor(y * k) * band.width + Math.floor(x * k)]
  }
  const channel = name => {
    if (name === 'ratio') {
      const vv = bands.vv, vh = bands.vh, out = new Float32Array(size * size)
      for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) out[y * size + x] = dB(at(vv, x, y)) - dB(at(vh, x, y))
      return { width: size, values: out, noData: null }
    }
    const b = bands[name]
    if (!s1) return b
    const out = new Float32Array(b.values.length)                       // S1: linear backscatter -> dB
    for (let i = 0; i < out.length; i++) out[i] = dB(b.values[i])
    return { width: b.width, values: out, noData: null }
  }
  const chans = view.use.map(channel), own = chans.map(c => stretch(c.values, c.noData))
  // S2 composites: one stretch for the three bands (reflectances on one scale), so colours keep their balance;
  // S1: each channel its own (VV, VH and their ratio are different quantities)
  const joint = !s1 && chans.length === 3 ? [Math.min(...own.map(r => r[0])), Math.max(...own.map(r => r[1]))] : null
  const ranges = joint ? own.map(() => joint) : own
  const canvas = new OffscreenCanvas(size, size), ctx = canvas.getContext('2d'), img = ctx.createImageData(size, size)
  const elevation = view.id === 'rgb:DEM'
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const o = (y * size + x) * 4
      const t = chans.map((c, k) => {
        const v = at(c, x, y), [lo, hi] = ranges[k]
        return Number.isFinite(v) && v !== c.noData ? Math.min(Math.max((v - lo) / (hi - lo), 0), 1) : NaN
      })
      if (t.some(Number.isNaN)) { img.data[o + 3] = 0; continue }
      const rgb = elevation ? terrain(t[0]) : chans.length === 1 ? [t[0] * 255, t[0] * 255, t[0] * 255] : t.map(v => v * 255)
      img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  const src = URL.createObjectURL(await canvas.convertToBlob({ type: 'image/png' }))
  const single = view.use.length === 1 && !elevation ? chans[0] : elevation ? bands.DEM : null
  stats.read = performance.now() - stats.t0
  return { src, size, values: single && { width: single.width, values: single.values, unit: s1 ? 'dB' : elevation ? 'm' : '' }, stats }
}
