// One Core sample from its image file on Hugging Face: the row group that holds it, thumbnail column only.
//
//   footprint row (part, parquet_row) -> <images>/part_NNNNN.parquet (one row group per sample)
//     -> footer (kept per file: 0.2-1.5 MB) -> that row group's thumbnail (and product_id where the file has it)
//     -> an object URL for <img>

import { compressors, countedFile, newStats, parquetMetadataAsync, parquetRead, TAIL } from './parquet.js'

const footers = new Map()                         // image file URL -> its metadata, after the first read

export const imageUrl = (info, part) => `${info.images}part_${String(part).padStart(5, '0')}.parquet`

/** The thumbnail of one sample: {src (object URL), productId, url, stats}. */
export async function loadThumbnail(info, sample) {
  const stats = newStats(), url = imageUrl(info, sample.part)
  const file = await countedFile(url, stats, footers.has(url) ? 8 : TAIL)           // 8 bytes: the size only
  if (!footers.has(url)) footers.set(url, await parquetMetadataAsync(file, { initialFetchSize: TAIL }))
  const metadata = footers.get(url), names = metadata.schema.slice(1).map(c => c.name)
  let rowStart = 0
  for (let g = 0; g < sample.parquet_row; g++) rowStart += Number(metadata.row_groups[g].num_rows)
  const columns = ['thumbnail', ...(names.includes('product_id') ? ['product_id'] : [])]
  let row
  await parquetRead({ file, metadata, compressors, columns, rowStart, rowEnd: rowStart + 1, rowFormat: 'object',
    utf8: false, onComplete(rows) { row = rows[0] } })                 // utf8: the thumbnail stays bytes
  if (!row?.thumbnail) throw new Error(`${url}: no thumbnail in row ${rowStart}`)
  stats.read = performance.now() - stats.t0
  const text = v => (v instanceof Uint8Array ? new TextDecoder().decode(v) : v ?? null)
  return { src: URL.createObjectURL(new Blob([row.thumbnail])), productId: text(row.product_id), url, stats }
}
