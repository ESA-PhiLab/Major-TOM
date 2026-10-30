// What the globe draws: map tiles made in the browser from the current field, and the Major TOM cell under a point.
//
//   MapLibre asks for tile z/x/y (Web Mercator, 256 px)
//     -> each pixel's longitude and latitude
//     -> the Major TOM cell it falls in (row from latitude, then column within that row's own width)
//     -> grid field: the cell's value -> colour scale;  image field: the preview pixel at that lon/lat
//     -> ImageBitmap handed back to MapLibre (no encoding: MapLibre uses decoded pixels as they are)

export const ROWS = 2004, COLS = 4008                  // the (row, column) array: rows -1002..1001, columns -2004..2003
const R_KM = 6378.137, D_KM = 10, DLAT = 180 / ROWS
const TILE = 256

/** Columns in grid row r (as majortom.cells: from the latitude of the row's lower edge). */
const nCols = r => Math.max(Math.ceil((2 * Math.PI * R_KM * Math.cos((r * DLAT * Math.PI) / 180)) / D_KM), 1)

/** Row and column of the cell under (lon, lat), lon in -180..180. */
function rowCol(lon, lat) {
  const row = Math.min(Math.max(Math.floor(lat / DLAT), -ROWS / 2), ROWS / 2 - 1)
  const n = nCols(row)
  const col = Math.min(Math.max(Math.floor(lon / (360 / n)), -Math.floor(n / 2)), n - 1 - Math.floor(n / 2))
  return [row, col, n]
}

/** Index of the cell under (lon, lat) in the (row, column) array, or at `level` in its 2^level-coarser block means
 *  (a Zarr overview); used for every pixel, so no objects. */
function cellIndex(lon, lat, level = 0) {
  const row = Math.min(Math.max(Math.floor(lat / DLAT), -ROWS / 2), ROWS / 2 - 1)
  const n = nCols(row)
  const col = Math.min(Math.max(Math.floor(lon / (360 / n)), -Math.floor(n / 2)), n - 1 - Math.floor(n / 2))
  if (!level) return (row + ROWS / 2) * COLS + col + COLS / 2
  return ((row + ROWS / 2) >> level) * Math.ceil(COLS / 2 ** level) + ((col + COLS / 2) >> level)
}

/** The cell under (lon, lat), for the cursor: row, column, name (as majortom.cells), index, bounds in degrees. */
export function cellAt(lon, lat) {
  lon = ((lon + 180) % 360 + 360) % 360 - 180
  const [row, col, n] = rowCol(lon, lat), dlon = 360 / n
  return {
    row, col, index: (row + ROWS / 2) * COLS + col + COLS / 2,
    name: `${Math.abs(row)}${row >= 0 ? 'U' : 'D'}_${Math.abs(col)}${col >= 0 ? 'R' : 'L'}`,
    bounds: [col * dlon, row * DLAT, (col + 1) * dlon, (row + 1) * DLAT],
  }
}

/** A field of values, one per cell, on the (row, column) array (or its block means at `level`); NaN where there
 *  is no cell or no value. */
export function gridField(values, scale, level = 0) {
  return {
    values, scale, level,
    valueAt: (lon, lat) => values[cellIndex(((lon + 180) % 360 + 360) % 360 - 180, lat, level)],
    pixel(lon, lat, out, o) {
      const v = values[cellIndex(lon, lat, level)]
      if (Number.isNaN(v)) { out[o + 3] = 0; return }
      const i = Math.round(scale.position(v) * 255) * 3
      out[o] = scale.lut[i]; out[o + 1] = scale.lut[i + 1]; out[o + 2] = scale.lut[i + 2]; out[o + 3] = 255
    },
  }
}

/** A field read from a preview image of the (row, column) array, north up (build_images.py): pixel (x, y) is the
 *  cell grid_col_r = x - 2004, grid_row_u = 1001 - y, so a point finds its pixel exactly as live data finds its value. */
export function imageField(rgba, scale) {
  const colour = (lon, lat, out, o) => {
    const k = cellIndex(lon, lat), i = ((ROWS - 1 - Math.floor(k / COLS)) * COLS + (k % COLS)) * 4
    out[o] = rgba[i]; out[o + 1] = rgba[i + 1]; out[o + 2] = rgba[i + 2]; out[o + 3] = rgba[i + 3]
  }
  return { scale, valueAt: () => undefined, pixel: colour }    // colours only: exact values come from the Parquet files
}

/** One 256 px tile of `field` at z/x/y. */
export async function renderTile(field, z, x, y) {
  const n = 2 ** z, out = new Uint8ClampedArray(TILE * TILE * 4)
  const lons = new Float64Array(TILE)
  for (let i = 0; i < TILE; i++) lons[i] = ((x + (i + 0.5) / TILE) / n) * 360 - 180
  for (let j = 0; j < TILE; j++) {
    const lat = (Math.atan(Math.sinh(Math.PI - (2 * Math.PI * (y + (j + 0.5) / TILE)) / n)) * 180) / Math.PI
    for (let i = 0; i < TILE; i++) field.pixel(lons[i], lat, out, (j * TILE + i) * 4)
  }
  return createImageBitmap(new ImageData(out, TILE, TILE))
}
