// Downloads of the selected sample: the bands exactly as stored (GeoTIFF bytes from the image file, not re-encoded).
//
//   one band   -> <cell>_<dataset>_<time>_<band>.tif
//   all bands  -> <cell>_<dataset>_<time>.zip: every band's .tif, the thumbnail, metadata.json (where it came from)
//   Python     -> a snippet that reads the same row group from Hugging Face (copied to the clipboard)
//   VIIRS      -> the tile's GeoTIFF as stored in its zip shard (already read); Python reads the same byte range

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
