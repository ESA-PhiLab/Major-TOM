// VIIRS Nighttime Light tiles (Major-TOM/Core-VIIRS-Nighttime-Light), read from their zip shards and coloured here.
//
//   footprint row (shard, offset, size) -> one range request: the tile's GeoTIFF, stored uncompressed in the zip
//     -> geotiff.js (zstd tiles: decoder registered below; geotiff.js 2.1.3 has none) -> radiance, nW/cm²/sr
//     -> log scale 0.1-100 in inferno, the Core Viewer's fixed scale, so tiles compare -> PNG
//   corners from the GeoTIFF's own bounds and UTM zone: VIIRS tiles are 1,056 px at 10 m on the v2 grid, not the
//   Core images' 10,680 m squares

import { addDecoder, BaseDecoder, fromArrayBuffer } from 'https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm'
import { decompress } from 'https://cdn.jsdelivr.net/npm/fzstd@0.1.1/+esm'
import proj4 from 'https://cdn.jsdelivr.net/npm/proj4@2.12.1/+esm'

class ZstdDecoder extends BaseDecoder {
  decodeBlock(buffer) {
    const out = decompress(new Uint8Array(buffer))
    return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength)
  }
}
addDecoder(50000, () => ZstdDecoder)

// matplotlib's inferno at nine stops
const INFERNO = [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85], [227, 89, 51], [249, 140, 10],
  [249, 201, 50], [252, 255, 164]]
const LO = Math.log10(0.1), HI = Math.log10(100)

function inferno(t) {
  const f = Math.min(Math.max(t, 0), 1) * (INFERNO.length - 1), k = Math.min(Math.floor(f), INFERNO.length - 2), u = f - k
  return INFERNO[k].map((c, i) => c + u * (INFERNO[k + 1][i] - c))
}

/** The sample's tile: {src (PNG object URL), size, values (radiance, for the cursor), corners, stats}. */
export async function loadViirs(info, sample, stats) {
  const r = await fetch(`${info.files}${sample.shard}`, { headers: { Range: `bytes=${sample.offset}-${sample.offset + sample.size - 1}` } })
  if (r.status !== 206) throw new Error(`${sample.shard}: HTTP ${r.status}, not a byte range`)
  const buffer = await r.arrayBuffer()
  stats.bytes += buffer.byteLength; stats.requests += 1
  const image = await (await fromArrayBuffer(buffer)).getImage(), [values] = await image.readRasters()
  const size = image.getWidth(), canvas = new OffscreenCanvas(size, size), ctx = canvas.getContext('2d')
  const img = ctx.createImageData(size, size)
  for (let i = 0; i < values.length; i++) {
    const v = values[i], o = i * 4
    if (!Number.isFinite(v)) continue                                    // transparent
    const [red, green, blue] = inferno((Math.log10(Math.max(v, 0.1)) - LO) / (HI - LO))
    img.data[o] = red; img.data[o + 1] = green; img.data[o + 2] = blue; img.data[o + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  const src = URL.createObjectURL(await canvas.convertToBlob({ type: 'image/png' }))
  const [west, south, east, north] = image.getBoundingBox(), epsg = image.geoKeys.ProjectedCSTypeGeoKey
  const utm = proj4(`+proj=utm +zone=${epsg % 100}${epsg > 32700 ? ' +south' : ''} +datum=WGS84 +units=m +no_defs`)
  const corners = [[west, north], [east, north], [east, south], [west, south]].map(p => utm.inverse(p))
  stats.read = performance.now() - stats.t0
  return { src, size, values: { width: size, values, unit: 'nW/cm²/sr' }, corners, stats }
}
