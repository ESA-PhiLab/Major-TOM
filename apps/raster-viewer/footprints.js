// Where a Core dataset's samples are, and where each one is stored: its footprint file (build_index.py).
//
//   footprints/<name>.parquet: one row per sample, sorted by grid_row_u, grid_col_r; page index
//     -> loadCoverage(): the two position columns -> index on the (row, column) array of every sample, ascending
//        -> mask (coverage tiles) and, for a click, a binary search for the cell's rows
//     -> samplesAt(): the cell's rows only, through the page index -> part, parquet_row, timestamp, ...

import { COLS, ROWS } from './tiles.js'
import { compressors, countedFile, newStats, parquetMetadataAsync, parquetRead } from './parquet.js'

export const DATASETS = fetch('footprints/datasets.json' + location.search).then(r => r.json())

const footprintUrl = name => new URL(`footprints/${name}.parquet`, location.href).href.split('?')[0] + location.search
const loaded = new Map()                         // name -> {file, metadata, index, mask}

/** A dataset's samples: index of each on the (row, column) array (ascending, the file's order) and a mask of the
 *  cells with a sample. onProgress(bytes so far, bytes expected). Read once per dataset. */
export function loadCoverage(name, onProgress = () => {}) {
  if (!loaded.has(name)) loaded.set(name, read(name, onProgress))
  return loaded.get(name)
}

async function read(name, onProgress) {
  const stats = newStats(), file = await countedFile(footprintUrl(name), stats)
  const metadata = await parquetMetadataAsync(file)
  let expected = 0
  for (const g of metadata.row_groups) {
    for (const { meta_data: m } of g.columns) {
      if (['grid_row_u', 'grid_col_r'].includes(m.path_in_schema[0])) expected += Number(m.total_compressed_size)
    }
  }
  const n = Number(metadata.num_rows), index = new Int32Array(n), mask = new Uint8Array(ROWS * COLS)
  const rows = new Int16Array(n), cols = new Int16Array(n)
  await parquetRead({
    file, metadata, compressors, columns: ['grid_row_u', 'grid_col_r'],
    onChunk({ columnName, columnData, rowStart }) {
      if (columnName === 'grid_row_u') rows.set(columnData, rowStart)
      else cols.set(columnData, rowStart)
      onProgress(stats.bytes, expected)
    },
  })
  for (let i = 0; i < n; i++) {
    index[i] = (rows[i] + ROWS / 2) * COLS + cols[i] + COLS / 2
    mask[index[i]] = 1
  }
  stats.read = performance.now() - stats.t0
  return { file, metadata, index, mask, stats, samples: n }
}

/** First position in the ascending `index` not below `k`. */
function lowerBound(index, k) {
  let lo = 0, hi = index.length
  while (lo < hi) { const mid = (lo + hi) >> 1; if (index[mid] < k) lo = mid + 1; else hi = mid }
  return lo
}

/** The samples of one cell (usually one; a few cells have two): their rows of the footprint file, read through
 *  the page index, as objects with that file's columns. */
export async function samplesAt(name, row, col) {
  const { file, metadata, index } = await loadCoverage(name), k = (row + ROWS / 2) * COLS + col + COLS / 2
  const first = lowerBound(index, k), last = lowerBound(index, k + 1)
  if (first === last) return []
  const columns = metadata.schema.slice(1).map(c => c.name)
  let out = []
  await parquetRead({ file, metadata, compressors, useOffsetIndex: true, rowStart: first, rowEnd: last, rowFormat: 'object',
    onComplete(rows) { out = rows } })
  return out.map(r => Object.fromEntries(columns.map(c => [c, typeof r[c] === 'bigint' ? Number(r[c]) : r[c]])))
}
