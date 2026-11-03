// zstd-compressed GeoTIFF tiles (VIIRS, AlphaEarth): geotiff.js 2.1.3 has no decoder for compression 50000, so one
// is registered here, on fzstd. Importing this module once is enough: geotiff.js keeps the decoder for every file.

import { addDecoder, BaseDecoder } from 'https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm'
import { decompress } from 'https://cdn.jsdelivr.net/npm/fzstd@0.1.1/+esm'

class ZstdDecoder extends BaseDecoder {
  decodeBlock(buffer) {
    const out = decompress(new Uint8Array(buffer))
    return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength)
  }
}
addDecoder(50000, () => ZstdDecoder)
