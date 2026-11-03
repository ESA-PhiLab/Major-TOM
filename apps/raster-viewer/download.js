// Downloads of the selected sample: the bands exactly as stored (GeoTIFF bytes from the image file, not re-encoded).
//
//   one band   -> <cell>_<dataset>_<time>_<band>.tif
//   all bands  -> <cell>_<dataset>_<time>.zip: every band's .tif, the thumbnail, metadata.json (where it came from)
//   Python     -> a snippet that reads the same row group from Hugging Face (copied to the clipboard)
//   VIIRS      -> the tile's GeoTIFF as stored in its zip shard (already read); Python reads the same byte range
//   AlphaEarth -> the shown image (.png); the shown band or all 64 at 10 m as a GeoTIFF written here; Python (rasterio)

import { zipSync } from 'https://cdn.jsdelivr.net/npm/fflate@0.8.2/+esm'
import { bandNames, readBands } from './bands.js'
import { imageUrl } from './samples.js'

const stem = (cell, name, sample) => [cell, name, sample.timestamp].filter(Boolean).join('_')

function save(blob, filename) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob); a.download = filename
  document.body.append(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000)
}

/** One band's GeoTIFF, as stored. */
export async function downloadBand(info, name, cell, sample, band) {
  const [bytes, stats] = await readBands(info, sample, [band])
  save(new Blob([bytes[band]], { type: 'image/tiff' }), `${stem(cell, name, sample)}_${band}.tif`)
  return stats
}

/** Every band, the thumbnail and metadata.json, in one zip (stored, not recompressed: the GeoTIFFs are LZW). */
export async function downloadAll(info, name, cell, sample, thumbnailSrc, productId, stats) {
  const bands = await bandNames(info, sample)
  const [bytes] = await readBands(info, sample, bands, stats)
  const base = stem(cell, name, sample), entries = {}
  for (const b of bands) entries[`${base}_${b}.tif`] = bytes[b]
  const thumb = new Uint8Array(await (await fetch(thumbnailSrc)).arrayBuffer())
  const png = thumb[0] === 0x89 && thumb[1] === 0x50
  entries[`${base}_thumbnail.${png ? 'png' : 'jpg'}`] = thumb
  const meta = { grid_cell: cell, dataset: `Major-TOM/Core-${name}`, product_id: productId, timestamp: sample.timestamp ?? null,
    cloud_cover: sample.cloud_cover ?? null, orbit_state: sample.orbit_state ?? null, epsg: sample.epsg,
    centre_lat: sample.centre_lat, centre_lon: sample.centre_lon, source_file: imageUrl(info, sample.part),
    row_group: sample.parquet_row, bands }
  entries['metadata.json'] = new TextEncoder().encode(JSON.stringify(meta, null, 1) + '\n')
  save(new Blob([zipSync(entries, { level: 0 })], { type: 'application/zip' }), `${base}.zip`)
  return stats
}

/** Python that reads the same sample from Hugging Face (one row group of the image file). */
export function pythonSnippet(info, name, cell, sample, bands) {
  return `import fsspec
import pyarrow.parquet as pq

url = "${imageUrl(info, sample.part)}"
with fsspec.open(url, "rb") as f:                      # reads only the parts of the file needed
    row = pq.ParquetFile(f).read_row_group(${sample.parquet_row}, columns=${JSON.stringify(bands)})
# ${cell}, Major-TOM/Core-${name}: each band is a GeoTIFF, e.g.
# from rasterio.io import MemoryFile; MemoryFile(row["${bands[0]}"][0].as_py()).open().read(1)
`
}

/** A VIIRS tile's GeoTIFF, as stored in its zip shard (the bytes the panel already read). */
export function downloadViirs(cell, sample, bytes) {
  save(new Blob([bytes], { type: 'image/tiff' }), `${cell}_VIIRS_${sample.year}.tif`)
}

/** Python that reads the same VIIRS tile: its byte range of the zip shard, opened as a GeoTIFF. */
export function viirsSnippet(info, cell, sample) {
  return `import fsspec
from rasterio.io import MemoryFile

url = "${info.files}${sample.shard}"
with fsspec.open(url, "rb") as f:                      # one range request: the tile is stored uncompressed in the zip
    f.seek(${sample.offset}); tile = f.read(${sample.size})
radiance = MemoryFile(tile).open().read(1)             # ${cell}, ${sample.year}: nW/cm²/sr, 10 m, float32
`
}

// ---- AlphaEarth: GeoTIFFs written here (int8, uncompressed, one strip per band), the shown image, Python ---------

const TIFF_BYTES = { 2: 1, 3: 2, 4: 4, 12: 8 }                       // ASCII, SHORT, LONG, DOUBLE

/** int8 bands (north up, NODATA -128) as a GeoTIFF in a UTM zone: west and north edges in metres, square pixels. */
export function int8GeoTiff(bands, size, { epsg, west, north, res }) {
  const n = bands.length, plane = size * size
  const entries = [[256, 4, [size]], [257, 4, [size]], [258, 3, Array(n).fill(8)], [259, 3, [1]], [262, 3, [1]],
    [273, 4, Array(n).fill(0)], [277, 3, [n]], [278, 4, [size]], [279, 4, Array(n).fill(plane)], [284, 3, [2]],
    ...(n > 1 ? [[338, 3, Array(n - 1).fill(0)]] : []), [339, 3, Array(n).fill(2)],
    [33550, 12, [res, res, 0]], [33922, 12, [0, 0, 0, west, north, 0]],
    [34735, 3, [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, epsg]],     // projected, pixel is area, EPSG
    [42113, 2, [...'-128'].map(c => c.charCodeAt(0)).concat(0)]]                  // GDAL_NODATA
  let at = 8 + 2 + 12 * entries.length + 4
  const places = entries.map(([, type, values]) => {
    const bytes = values.length * TIFF_BYTES[type]
    if (bytes <= 4) return null
    const here = at; at += bytes + (bytes % 2); return here
  })
  entries[5][2] = bands.map((_, b) => at + b * plane)                               // StripOffsets: after the values
  const buffer = new ArrayBuffer(at + n * plane), view = new DataView(buffer), bytes = new Uint8Array(buffer)
  const put = (offset, type, v) => {
    if (type === 2) view.setUint8(offset, v)
    else if (type === 3) view.setUint16(offset, v, true)
    else if (type === 4) view.setUint32(offset, v, true)
    else view.setFloat64(offset, v, true)
  }
  view.setUint16(0, 0x4949, true); view.setUint16(2, 42, true); view.setUint32(4, 8, true)
  view.setUint16(8, entries.length, true)
  entries.forEach(([tag, type, values], k) => {
    const e = 10 + 12 * k
    view.setUint16(e, tag, true); view.setUint16(e + 2, type, true); view.setUint32(e + 4, values.length, true)
    const base = places[k] ?? e + 8
    if (places[k] !== null) view.setUint32(e + 8, base, true)
    values.forEach((v, i) => put(base + i * TIFF_BYTES[type], type, v))
  })
  bands.forEach((band, b) => bytes.set(new Uint8Array(band.buffer, band.byteOffset, plane), at + b * plane))
  return new Blob([buffer], { type: 'image/tiff' })
}

/** The shown image (a PNG object URL), saved as drawn. */
export async function downloadImage(src, filename) {
  save(await (await fetch(src)).blob(), filename)
}

/** int8 bands of the AlphaEarth window as a GeoTIFF. */
export function downloadAefTiff(window, filename) {
  save(int8GeoTiff(window.bands, window.size, { epsg: window.cell.epsg, west: window.west, north: window.north, res: window.res }),
    filename)
}

/** Python that reads the same AlphaEarth window: each tile's part (a windowed read; the files are stored south-up),
 *  joined, north up, on the 10 m grid the app uses. */
export function aefSnippet(cellName, sample, cell) {
  const west = Math.round((cell.cx - 5340) / 10) * 10, north = Math.round((cell.cy - 5340) / 10) * 10 + 10680
  return `import numpy as np
import rasterio
from rasterio.windows import Window

urls = ${JSON.stringify(cell.urls, null, 4).replace(/\n\]$/, ',\n]')}
west, north = ${west}, ${north}   # ${cellName}, EPSG:${cell.epsg}: 1,068 x 1,068 pixels of 10 m
embedding = np.full((64, 1068, 1068), -128, np.int8)      # AlphaEarth ${sample.year} (tge-labs/aef, CC-BY 4.0)
for url in urls:                                          # every tile under the cell
    with rasterio.open(url) as src:
        t = src.transform                                 # t.e > 0: stored south-up
        edge = north - 10680 if t.e > 0 else north
        win = Window(round((west - t.c) / t.a), round((edge - t.f) / t.e), 1068, 1068)
        part = src.read(window=win, boundless=True, fill_value=-128)
        if t.e > 0:
            part = part[:, ::-1]                          # north up
        embedding = np.where(embedding == -128, part, embedding)
`
}
