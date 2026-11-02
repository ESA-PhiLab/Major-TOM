// Remote Parquet files over HTTP range requests, with the bytes, requests and network time each read costs
// (as climate-globe/data.js).

import { parquetMetadataAsync } from 'https://cdn.jsdelivr.net/npm/hyparquet@1.31.4/+esm'
export { parquetMetadataAsync, parquetRead } from 'https://cdn.jsdelivr.net/npm/hyparquet@1.31.4/+esm'
export { compressors } from 'https://cdn.jsdelivr.net/npm/hyparquet-compressors@1.1.2/+esm'

export const TAIL = 131072      // a file's first request: its last 128 KB, which hold most footers here

export const newStats = () => ({ bytes: 0, requests: 0, network: 0, active: 0, t0: performance.now() })

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
export async function countedFile(url, stats, tailSize = TAIL) {
  const [r, tail] = await rangeGet(url, `-${tailSize}`, stats)
  const range = r.headers.get('content-range')                                    // "bytes a-b/size"; absent if whole
  const byteLength = range ? Number(range.split('/')[1]) : tail.byteLength
  const tailStart = byteLength - tail.byteLength
  const file = {
    byteLength, stats,                                   // stats: the read being counted, set again by openFile
    async slice(start, end = byteLength) {
      if (start >= tailStart) return tail.slice(start - tailStart, end - tailStart)
      return (await rangeGet(url, `${start}-${end - 1}`, file.stats))[1]
    },
  }
  return file
}

const opened = new Map()                       // URL -> {file, metadata}: a file's footer is read once per page

/** A remote file and its metadata, opened once; later reads are counted in `stats`. */
export async function openFile(url, stats) {
  if (!opened.has(url)) {
    opened.set(url, (async () => {
      const file = await countedFile(url, stats)
      return { file, metadata: await parquetMetadataAsync(file, { initialFetchSize: TAIL }) }
    })().catch(e => { opened.delete(url); throw e }))
  }
  const found = await opened.get(url)
  found.file.stats = stats
  return found
}
