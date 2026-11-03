// The page: a MapLibre globe of where a Core dataset has samples, and the image of the sample in a clicked cell.
//
//   dataset (panel) -> footprints/<name>.parquet, two position columns (footprints.js) -> mask on the cell array
//     -> coverage tiles drawn in the browser (tiles.js) -> globe
//   datasets with a kind have no thumbnail column: VIIRS tiles are read and coloured (viirs.js); AlphaEarth windows
//     are read from the yearly COG and shown as a PCA of the 64 bands (aef.js), one choice button per year
//   click -> cell -> its samples (footprint rows, page index) -> image file on Hugging Face, one row group,
//     thumbnail column (samples.js) -> panel
//   cursor -> cell under it -> outline + name, position, whether it has a sample
//   the shown image (the stored thumbnail, or a band or composite decoded from the image file, bands.js)
//     -> full window (zoom, pan, value under the cursor) and on the map over its footprint (corners from its UTM centre)
//   the sample's acquisition -> Core-Weather on source.coop, ±24 h and ±30 d (weather.js) -> charts in the panel
//     -> downloads of the bands as stored (download.js)

import * as maplibregl from 'https://cdn.jsdelivr.net/npm/maplibre-gl@6.14.0/dist/maplibre-gl.mjs'
import { feature } from 'https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/+esm'
import proj4 from 'https://cdn.jsdelivr.net/npm/proj4@2.12.1/+esm'
import { DATASETS, loadCoverage, samplesAt } from './footprints.js'
import { placeLabels } from './labels.js'
import { imageUrl, loadThumbnail } from './samples.js'
import { bandNames, render, views } from './bands.js'
import { newStats } from './parquet.js'
import { loadWeather, weatherRows } from './weather.js'
import { loadViirs } from './viirs.js'
import { aefViews, renderAef } from './aef.js'
import { downloadAll, downloadBand, downloadViirs, pythonSnippet, viirsSnippet } from './download.js'
import { cellAt, cellIndex, renderTile } from './tiles.js'

const $ = id => document.getElementById(id)
const COLOURS = { S2L2A: [57, 181, 74], S2L1C: [0, 113, 188], S1RTC: [241, 90, 36], DEM: [180, 140, 90], VIIRS: [126, 87, 194], AEF: [32, 150, 150] }

const state = { name: null, info: null, datasets: null, coverage: null, version: 0, loading: 0 }

// ---- map ---------------------------------------------------------------------------------------------------

/** Coverage of the shown dataset: its colour where a cell has a sample, transparent elsewhere. */
const field = {
  pixel(lon, lat, out, o) {
    const mask = state.coverage?.mask
    if (!mask || !mask[cellIndex(lon, lat)]) { out[o + 3] = 0; return }
    const [r, g, b] = COLOURS[state.name] ?? [57, 181, 74]
    out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 130     // coastlines show through
  },
}
maplibregl.addProtocol('mt', async params => {
  const [z, x, y] = params.url.split('/').slice(-3).map(Number)
  return { data: await renderTile(field, z, x, y) }
})

const map = new maplibregl.Map({
  container: 'map', center: [12, 22], zoom: innerWidth < 820 ? 1.3 : 2.25, attributionControl: false, maxZoom: 13,
  style: {
    version: 8, projection: { type: 'globe' },
    sky: { 'atmosphere-blend': ['interpolate', ['linear'], ['zoom'], 0, 0.6, 4, 0.3, 7, 0] },
    sources: {
      field: { type: 'raster', tiles: ['mt://0/{z}/{x}/{y}'], tileSize: 256, maxzoom: 9 },
      land: { type: 'geojson', data: { type: 'FeatureCollection', features: [] } },
      coast: { type: 'geojson', data: { type: 'FeatureCollection', features: [] } },
      cell: { type: 'geojson', data: { type: 'FeatureCollection', features: [] } },
      selected: { type: 'geojson', data: { type: 'FeatureCollection', features: [] } },
    },
    layers: [
      { id: 'ocean', type: 'background', paint: { 'background-color': '#2a2d33' } },
      { id: 'land', type: 'fill', source: 'land', paint: { 'fill-color': '#3b3f47', 'fill-antialias': false } },
      { id: 'field', type: 'raster', source: 'field', paint: { 'raster-resampling': 'nearest', 'raster-fade-duration': 0 } },
      { id: 'coast', type: 'line', source: 'coast', paint: { 'line-color': 'rgba(20,20,24,0.55)', 'line-width': 0.6 } },
      { id: 'cell', type: 'line', source: 'cell', paint: { 'line-color': '#f15a24', 'line-width': 1.25, 'line-opacity': 0.8 } },
      { id: 'selected', type: 'line', source: 'selected', paint: { 'line-color': '#f15a24', 'line-width': 2.5 } },
    ],
  },
})
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
const places = placeLabels(map, $('places'))
$('show-places').addEventListener('change', e => places.show(e.target.checked))
map.addControl(new maplibregl.AttributionControl({ compact: true,
  customAttribution: 'Coastlines and places: Natural Earth · Data: <a href="https://huggingface.co/Major-TOM">Major TOM Core</a>' }))

map.on('load', async () => {
  const topo = await (await fetch('https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-50m.json')).json()
  map.getSource('coast').setData(coastLines(feature(topo, topo.objects.land)))
  map.getSource('land').setData(continuousRings(feature(topo, topo.objects.land)))
  const datasets = await DATASETS
  state.datasets = datasets
  buildDatasetList(datasets)
  setDataset(Object.keys(datasets)[0], datasets)
})

/** Land polygons whose rings step across the date line made continuous (as climate-globe/app.js). */
function continuousRings(land) {
  const unwrap = ring => {
    if (ring.some(([, y]) => y <= -89.99)) return ring
    const out = [ring[0]]
    for (let i = 1, shift = 0; i < ring.length; i++) {
      const dx = ring[i][0] + shift - out[i - 1][0]
      if (dx > 180) shift -= 360; else if (dx < -180) shift += 360
      out.push([ring[i][0] + shift, ring[i][1]])
    }
    return out
  }
  for (const f of land.features) {
    const g = f.geometry
    g.coordinates = g.type === 'Polygon' ? g.coordinates.map(unwrap) : g.coordinates.map(p => p.map(unwrap))
  }
  return land
}

/** Coastlines from the land polygons, without the edges that are not coast (as climate-globe/app.js). */
function coastLines(land) {
  const artificial = ([x1, y1], [x2, y2]) => (Math.abs(x1) >= 179.99 && Math.abs(x2) >= 179.99) ||
    (y1 <= -89.99 && y2 <= -89.99) || Math.abs(x2 - x1) > 180
  const lines = []
  for (const f of land.features) {
    const polygons = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates
    for (const ring of polygons.flat()) {
      let line = [ring[0]]
      for (let i = 1; i < ring.length; i++) {
        if (artificial(ring[i - 1], ring[i])) { if (line.length > 1) lines.push(line); line = [ring[i]] }
        else line.push(ring[i])
      }
      if (line.length > 1) lines.push(line)
    }
  }
  return { type: 'Feature', geometry: { type: 'MultiLineString', coordinates: lines } }
}

// ---- cursor ------------------------------------------------------------------------------------------------

let cursor = null

/** Whether a screen point lies on the globe (not the space around it). */
function onGlobe(point, lngLat) {
  if (typeof map.transform?.isPointOnMapSurface === 'function') return map.transform.isPointOnMapSurface(point)
  const back = map.project(lngLat)
  return Math.hypot(back.x - point.x, back.y - point.y) < 2
}
map.on('mousemove', e => {
  if (onGlobe(e.point, e.lngLat)) { cursor = e.lngLat; describe() } else clearCursor()
})
window.globe = map                                         // for automated checks
window.pickState = () => pick

const outline = ([w, s, e, n]) => ({ type: 'Feature', geometry: { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } })

function describe() {
  if (!cursor) return
  const cell = cellAt(cursor.lng, cursor.lat)
  map.getSource('cell').setData(outline(cell.bounds))
  $('cursor-cell').textContent = cell.name
  $('cursor-pos').textContent = `${cursor.lat.toFixed(2)}°, ${cursor.lng.toFixed(2)}°`
  $('cursor-value').textContent = !state.coverage ? '–' : state.coverage.mask[cell.index] ? 'yes, click to load' : 'none'
}
function clearCursor() {
  cursor = null
  map.getSource('cell').setData({ type: 'FeatureCollection', features: [] })
  for (const id of ['cursor-cell', 'cursor-pos', 'cursor-value']) $(id).textContent = '–'
}
map.getCanvas().addEventListener('mouseleave', clearCursor)

// ---- loading indicator (as climate-globe/app.js, with the Major TOM logo) -------------------------------------------------------------

const bars = [...document.querySelectorAll('#loader polygon')]                  // the logo's 4 panels, left to right
const still = matchMedia('(prefers-reduced-motion: reduce)').matches
let loaderTimer = null, animation = null, tick = 0

function animateFlag() {
  const n = bars.length, k = tick++ % (2 * n)                  // first n ticks light the panels, next n clear them
  if (k < n) bars[k].classList.add('on')
  else bars[k - n].classList.remove('on')
}
function startLoader(text) {
  for (const b of bars) b.classList.remove('on')
  tick = 0
  $('loader-phase').textContent = 'Loading'
  $('loader-text').textContent = text
  $('loader-bytes').textContent = ''
  clearTimeout(loaderTimer); clearInterval(animation)
  loaderTimer = setTimeout(() => {                                               // fast and cached loads: no flash
    $('loader').hidden = false
    if (still) bars.slice(0, 2).forEach(b => b.classList.add('on'))
    else animation = setInterval(animateFlag, 180)
  }, 150)
}
function loaderProgress(bytes, expected) {
  $('loader-bytes').textContent = expected ? `${(bytes / 1e6).toFixed(1)} of ${(Math.max(expected, bytes) / 1e6).toFixed(1)} MB`
    : `${(bytes / 1e6).toFixed(1)} MB`
}
function stopLoader() {
  clearTimeout(loaderTimer); clearInterval(animation)
  $('loader').hidden = true
}

// ---- dataset -----------------------------------------------------------------------------------------------

const kb = bytes => (bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.round(bytes / 1e3)} KB`)
const secs = ms => `${(ms / 1000).toFixed(1)} s`
const reqs = n => `${n} request${n === 1 ? '' : 's'}`

function buildDatasetList(datasets) {
  $('datasets').replaceChildren(...Object.entries(datasets).map(([name, d]) => {
    const label = document.createElement('label')
    label.className = 'option'
    label.innerHTML = `<input type="radio" name="dataset" value="${name}"><span><b>${d.label}</b>` +
      `<small>Core-${name} · ${d.samples.toLocaleString('en')} samples</small></span>`
    label.querySelector('input').addEventListener('change', () => setDataset(name, datasets))
    return label
  }))
}

async function setDataset(name, datasets) {
  const ticket = ++state.loading, info = datasets[name]
  state.name = name; state.info = info; state.coverage = null
  document.querySelector(`input[name=dataset][value="${name}"]`).checked = true
  const link = info.link ?? `https://huggingface.co/datasets/Major-TOM/Core-${name}`
  $('dataset-link').href = link
  $('dataset-link').textContent = link.replace('https://', '')
  $('status').textContent = ''
  startLoader(`${info.label} · footprints of ${info.samples.toLocaleString('en')} samples`)
  let coverage
  try {
    coverage = await loadCoverage(name, (bytes, expected) => { if (ticket === state.loading) loaderProgress(bytes, expected) })
  } catch (error) {
    if (ticket === state.loading) { stopLoader(); $('status').textContent = `Could not load: ${error.message}` }
    return
  }
  if (ticket !== state.loading) return
  state.coverage = coverage
  $('loader-phase').textContent = 'Drawing'
  const t0 = performance.now()
  map.getSource('field').setTiles([`mt://${++state.version}/{z}/{x}/{y}`])
  describe()
  if (pick.cell) pickCell()
  map.once('idle', () => {
    if (ticket !== state.loading) return
    $('stat-footprints').textContent = `${kb(coverage.stats.bytes)} in ${reqs(coverage.stats.requests)} · ${secs(coverage.stats.read)}`
    $('stat-samples').textContent = coverage.samples.toLocaleString('en')
    $('stat-draw').textContent = secs(performance.now() - t0)
    document.body.dataset.loaded = name                     // for automated checks
    stopLoader()
    prefetchFootprints()
  })
}

/** The other datasets' footprints, one after another, once the shown one is drawn: switching is then instant. */
async function prefetchFootprints() {
  for (const name of Object.keys(state.datasets)) {
    try { await loadCoverage(name) } catch { /* loaded again, with its error shown, when picked */ }
  }
}

// ---- selected cell -----------------------------------------------------------------------------------------
//   click -> cell -> its samples (footprint rows) -> buttons when there are several -> thumbnail of the chosen one
//   then, in the background, the same cell in the other datasets: samples and first thumbnail, one dataset at a time
//   lookups and thumbnails are kept (promises, so a load under way is joined, not repeated)

const lookups = new Map()                                 // "<dataset>/<row>/<col>" -> samples
const thumbs = new Map()                                  // "<dataset>/<part>/<parquet_row>" -> thumbnail

function samplesOf(name, row, col) {
  const key = `${name}/${row}/${col}`
  if (!lookups.has(key)) lookups.set(key, samplesAt(name, row, col).then(rows => byYear(name, rows)).catch(e => { lookups.delete(key); throw e }))
  return lookups.get(key)
}

/** AlphaEarth: the cell's one footprint row as one sample per year (cellYear: its Sentinel-2 year, the default). */
function byYear(name, rows) {
  const info = state.datasets[name]
  if (info.kind !== 'aef' || !rows.length) return rows
  const [first, last] = info.years
  return Array.from({ length: last - first + 1 }, (_, k) => ({ ...rows[0], year: first + k, cellYear: rows[0].year }))
}

/** A sample's key, unique within its dataset. */
const sampleKey = s => (s.part !== undefined ? `${s.part}/${s.parquet_row}` : s.shard ? `${s.shard}/${s.offset}`
  : `${s.grid_row_u}/${s.grid_col_r}/${s.year}`)

/** The first image of a sample, read once: [promise, fresh, stats]. Thumbnail (Core), tile (VIIRS) or PCA (AlphaEarth). */
function thumbnailOf(name, sample) {
  const info = state.datasets[name], key = `${name}/${sampleKey(sample)}`, fresh = !thumbs.has(key)
  if (fresh) {
    const stats = newStats()
    const load = info.kind === 'viirs' ? loadViirs(info, sample, stats) : info.kind === 'aef' ? renderAef(info, sample, 'pca', stats, view => stats.onPreview?.(view))
      : loadThumbnail(info, sample)
    thumbs.set(key, { promise: load.catch(e => { thumbs.delete(key); throw e }), stats })
  }
  return [thumbs.get(key).promise, fresh, thumbs.get(key).stats]
}

/** The selected cell in every other dataset, in the panel's order, after the shown one. */
async function prefetchCell(cell) {
  for (const name of Object.keys(state.datasets)) {
    if (pick.cell !== cell) return                         // another cell was picked
    if (name === state.name || state.datasets[name].kind === 'aef') continue   // AlphaEarth: tens of MB, only on request
    try {
      const samples = await samplesOf(name, cell.row, cell.col)
      if (samples.length) await thumbnailOf(name, samples[0])[0]
    } catch { /* loaded again, with its error shown, when that dataset is shown */ }
  }
}

const pick = { cell: null, samples: [], chosen: 0, thumbnail: null, shown: null, views: [], bands: [], ticket: 0, viewTicket: 0 }

map.on('click', e => {
  if (!onGlobe(e.point, e.lngLat)) return
  pick.cell = { ...cellAt(e.lngLat.lng, e.lngLat.lat), lng: e.lngLat.lng, lat: e.lngLat.lat }
  map.getSource('selected').setData(outline(pick.cell.bounds))
  $('pick').hidden = false
  $('pick-cell').textContent = pick.cell.name
  $('pick-pos').textContent = `${pick.cell.lat.toFixed(2)}°, ${pick.cell.lng.toFixed(2)}°`
  $('pick').scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  pickCell()
})
$('pick-clear').addEventListener('click', () => {
  pick.cell = null; pick.ticket++; $('pick').hidden = true
  map.getSource('selected').setData({ type: 'FeatureCollection', features: [] })
  showOnMap()
})

/** The selected cell in the shown dataset: its samples, then the first one's thumbnail. */
async function pickCell() {
  const ticket = ++pick.ticket, { row, col } = pick.cell, name = state.name
  pick.samples = []; pick.thumbnail = null; pick.shown = null; pick.viewTicket++
  showThumbnail(null, 'Looking up the cell…'); showOnMap()
  $('choices').replaceChildren(); $('pick-meta').replaceChildren(); $('pick-stats').textContent = ''; $('weather').hidden = true
  let samples
  try {
    samples = await samplesOf(name, row, col)
  } catch (error) {
    if (ticket === pick.ticket) showThumbnail(null, `Could not load: ${error.message}`)
    return
  }
  if (ticket !== pick.ticket) return
  pick.samples = samples
  if (!samples.length) { showThumbnail(null, `No ${state.info.label} sample in this cell`); prefetchCell(pick.cell); return }
  if (samples.length > 1) {
    $('choices').replaceChildren(...samples.map((s, i) => {
      const b = document.createElement('button')
      b.textContent = s.timestamp ? when(s.timestamp) : s.year ? String(s.year) : `sample ${i + 1}`
      b.addEventListener('click', () => choose(i))
      return b
    }))
  }
  choose(Math.max(0, samples.findIndex(s => s.cellYear !== undefined && s.year === s.cellYear)))
}

async function choose(i) {
  const ticket = ++pick.ticket, sample = pick.samples[i], info = state.info, name = state.name
  pick.chosen = i; pick.shown = null; pick.bands = []; pick.views = []; pick.viewTicket++
  for (const [k, b] of [...$('choices').children].entries()) b.setAttribute('aria-pressed', String(k === i))
  meta(sample, null, info)
  showWeather(sample, ticket)
  showThumbnail(null, info.kind === 'aef' ? 'Reading the 64 bands…' : 'Loading the thumbnail…')
  try {
    const [loading, fresh, stats] = thumbnailOf(state.name, sample)
    const stop = fresh && info.kind === 'aef' ? followLoad(`AlphaEarth ${sample.year} · ${pick.cell.name} · 64 bands`, stats) : () => {}
    let unbadge = () => {}
    stats.onPreview = view => {                                        // AlphaEarth: the 160 m PCA while 40 m loads
      stop()
      if (ticket !== pick.ticket) return
      pick.shown = { ...view, sample: { ...sample, corners: view.corners } }
      showThumbnail(view.src); showOnMap()
      unbadge = sharpening('Sharpening to 40 m', stats)
      $('pick-stats').textContent = `${view.label} · sharpening to 40 m…`
    }
    const t = await loading.finally(() => { stop(); unbadge() })
    if (ticket !== pick.ticket) return
    const label = { viirs: 'Radiance', aef: aefViews()[0].label }[info.kind] ?? 'Thumbnail'
    const placed = t.corners ? { ...sample, corners: t.corners } : sample
    pick.thumbnail = { ...t, sample: placed }
    pick.shown = { src: t.src, values: t.values ?? null, size: t.size, sample: placed, label }
    showThumbnail(t.src)
    showOnMap()
    meta(sample, t.productId, info)
    if (info.kind === 'aef') aefMenu(label)
    else if (info.kind) updateDownloads()                               // no band menu: the tile is one band
    else buildViews(name, info, sample, ticket)
    $('pick-stats').textContent = `${label}: ${kb(t.stats.bytes)} in ${reqs(t.stats.requests)} · ${secs(t.stats.read)}` +
      (fresh ? '' : ' (loaded before)')
    prefetchCell(pick.cell)
  } catch (error) {
    if (ticket === pick.ticket) showThumbnail(null, `Could not load: ${error.message}`)
  }
}

/** The badge over a preview while its sharper version loads: the share of its bytes so far. Returns the function
 *  that ends it. */
function sharpening(text, stats) {
  const base = stats.bytes, badge = $('thumb-badge')
  const update = () => {
    const share = stats.expected > base ? Math.min(99, Math.round(100 * (stats.bytes - base) / (stats.expected - base))) : 0
    $('thumb-badge-text').textContent = `${text} · ${share}%`
  }
  update(); badge.hidden = false
  const timer = setInterval(update, 150)
  return () => { clearInterval(timer); badge.hidden = true }
}

function showThumbnail(src, note = '') {
  $('thumb-badge').hidden = true                                        // a new image: no sharpening under way for it
  $('thumb').hidden = !src
  $('image-actions').hidden = !src
  $('download-actions').hidden = !src
  if (!src) $('view-row').hidden = true
  if (src) $('thumb').src = src
  $('thumb-note').textContent = src ? '' : note
}

// ---- the shown image on the map -------------------------------------------------------------------------------
// Every Core image is 10,680 m square in its UTM zone, centred on the sample's centre_lat, centre_lon (checked
// against the GeoTIFF bounds), north up: its corners are the centre +- 5,340 m, back in longitude and latitude.

const HALF = 5340

/** Corners of a sample's image, [lon, lat], from top left clockwise (as MapLibre's image source wants them). */
function corners(s) {
  if (s.corners) return s.corners                                      // VIIRS: from the GeoTIFF's own bounds
  const zone = s.epsg % 100, south = s.epsg > 32700
  const utm = proj4(`+proj=utm +zone=${zone}${south ? ' +south' : ''} +datum=WGS84 +units=m +no_defs`)
  const [x, y] = utm.forward([s.centre_lon, s.centre_lat])
  return [[x - HALF, y + HALF], [x + HALF, y + HALF], [x + HALF, y - HALF], [x - HALF, y - HALF]].map(p => utm.inverse(p))
}

/** The shown image over its footprint, opaque, while a cell is selected. */
function showOnMap() {
  const t = pick.shown, on = t && pick.cell
  if (map.getLayer('sample')) map.removeLayer('sample')
  if (map.getSource('sample')) map.removeSource('sample')
  if (!on) return
  map.addSource('sample', { type: 'image', url: t.src, coordinates: corners(t.sample) })
  map.addLayer({ id: 'sample', type: 'raster', source: 'sample',
    paint: { 'raster-opacity': 1, 'raster-fade-duration': 0 } }, 'coast')
}
window.sampleCorners = () => (pick.shown ? corners(pick.shown.sample) : null)             // for automated checks

// ---- Core-Weather at the acquisition --------------------------------------------------------------------------

const WEATHER_DATASETS = ['S2L2A', 'S2L1C', 'S1RTC']                   // Core-Weather has rows for these three

/** The sample's hourly and daily weather in the panel. Datasets without an acquisition time of their own (DEM, VIIRS
 *  annual medians) show the weather at the cell's Sentinel-2 L2A acquisition, and say so. */
async function showWeather(sample, ticket) {
  let at = sample, from = ''
  if (!(sample.timestamp && WEATHER_DATASETS.includes(state.name))) {
    const s2 = await samplesOf('S2L2A', pick.cell.row, pick.cell.col).catch(() => [])
    if (ticket !== pick.ticket) return
    at = s2[0]; from = at?.timestamp ? `At the cell's Sentinel-2 L2A acquisition, ${when(at.timestamp)}` : ''
  }
  $('weather').hidden = !at?.timestamp
  if ($('weather').hidden) return
  $('wx-from').textContent = from; $('wx-from').hidden = !from
  $('wx-hourly').replaceChildren(); $('wx-daily').replaceChildren(); $('wx-stats').textContent = 'Loading Core-Weather…'
  try {
    const w = await loadWeather(pick.cell.name, at.timestamp)
    if (ticket !== pick.ticket) return
    $('wx-hourly').replaceChildren(...weatherRows(w.hourly, 'hourly'))
    $('wx-daily').replaceChildren(...weatherRows(w.daily, 'daily'))
    $('wx-stats').textContent = (w.hourly || w.daily ? '' : 'No Core-Weather row for this acquisition · ') +
      `Core-Weather: ${kb(w.stats.bytes)} in ${reqs(w.stats.requests)} · ${secs(w.stats.read)}`
  } catch (error) {
    if (ticket === pick.ticket) $('wx-stats').textContent = `Could not load Core-Weather: ${error.message}`
  }
}

// ---- views: the stored thumbnail, composites and single bands ------------------------------------------------
//   the menu lists the image file's band columns; a chosen view is decoded once and kept (renders)

const renders = new Map()                                 // "<dataset>/<part>/<row>/<view>" -> rendered view

async function buildViews(name, info, sample, ticket) {
  try {
    const bands = await bandNames(info, sample)
    if (ticket !== pick.ticket) return
    pick.bands = bands; pick.views = views(name, bands)
    $('view').replaceChildren(new Option('Thumbnail (stored)', 'thumbnail'), ...pick.views.map(v => new Option(v.label, v.id)))
    $('view').value = 'thumbnail'; $('view-row').hidden = false
    updateDownloads()
  } catch { /* no menu: the thumbnail stays */ }
}

/** AlphaEarth's View menu: the PCA shown (40 m), PCA at 10 m, each band. No downloads yet: they read Core files. */
function aefMenu(label) {
  pick.views = aefViews().slice(1)
  $('view').replaceChildren(new Option(label, 'thumbnail'), ...pick.views.map(v => new Option(v.label, v.id)))
  $('view').value = 'thumbnail'; $('view-row').hidden = false
  $('download-actions').hidden = true
}

$('view').addEventListener('change', () => showView($('view').value))

async function showView(id) {
  const t = pick.thumbnail, ticket = ++pick.viewTicket
  if (!t) return
  if (id === 'thumbnail') {
    pick.shown = { src: t.src, values: t.values ?? null, size: t.size, sample: t.sample, label: $('view').options[0].text }
    refreshShown(); return
  }
  const view = pick.views.find(v => v.id === id), key = `${state.name}/${sampleKey(t.sample)}/${id}`
  $('pick-stats').textContent = `Loading ${view.label}…`
  const fresh = !renders.has(key), stats = newStats()
  if (fresh) {
    const job = state.info.kind === 'aef' ? renderAef(state.info, t.sample, id, stats) : render(state.info, t.sample, view, stats)
    renders.set(key, job.catch(e => { renders.delete(key); throw e }))
  }
  const stop = fresh ? followLoad(`${view.label} · ${pick.cell.name}`, stats) : () => {}
  try {
    const r = await renders.get(key).finally(stop)
    if (ticket !== pick.viewTicket) return
    pick.shown = { ...r, sample: t.sample, label: view.label }
    refreshShown()
    $('pick-stats').textContent = `${view.label}: ${kb(r.stats.bytes)} in ${reqs(r.stats.requests)} · ${secs(r.stats.read)}` +
      (fresh ? '' : ' (loaded before)')
  } catch (error) {
    if (ticket === pick.viewTicket) $('pick-stats').textContent = `Could not load ${view.label}: ${error.message}`
  }
}

/** The loading flag over the map while a band read runs: bytes of the expected total, then its phase (decoding).
 *  Returns the function that ends it. */
function followLoad(text, stats) {
  startLoader(text)
  const timer = setInterval(() => {
    loaderProgress(stats.bytes, stats.expected)
    if (stats.phase) $('loader-phase').textContent = stats.phase
  }, 120)
  return () => { clearInterval(timer); stopLoader() }
}

function refreshShown() {
  showThumbnail(pick.shown.src)
  showOnMap()
  if (!$('lightbox').hidden) openImage()
  updateDownloads()
}

// ---- downloads ---------------------------------------------------------------------------------------------

const shownBand = () => (pick.shown && pick.bands.includes(pick.shown.label) ? pick.shown.label : null)

function updateDownloads() {
  const viirs = state.info?.kind === 'viirs'
  $('dl-all').hidden = viirs
  if (viirs) {
    $('dl-band').disabled = false; $('dl-band').textContent = 'Download tile (.tif)'
    $('dl-band').title = 'The tile as stored: a GeoTIFF, radiance in nW/cm²/sr'
    return
  }
  const band = shownBand()
  $('dl-band').disabled = !band
  $('dl-band').textContent = band ? `Download ${band} (.tif)` : 'Download band (.tif)'
  $('dl-band').title = band ? `${band} as stored: a GeoTIFF` : 'Choose a single band under View'
  $('dl-all').disabled = !pick.bands.length
}

async function downloading(button, job) {
  const label = button.textContent
  button.disabled = true; button.textContent = 'Preparing…'
  try {
    const stats = await job()
    $('pick-stats').textContent = `Downloaded: ${kb(stats.bytes)} read in ${reqs(stats.requests)} · ${secs(stats.read)}`
  } catch (error) {
    $('pick-stats').textContent = `Could not download: ${error.message}`
  } finally { button.textContent = label; updateDownloads() }
}

$('dl-band').addEventListener('click', () => {
  if (state.info.kind === 'viirs') {
    downloadViirs(pick.cell.name, pick.thumbnail.sample, pick.thumbnail.bytes)
    $('pick-stats').textContent = `Downloaded: ${kb(pick.thumbnail.bytes.length)}, read before`
    return
  }
  downloading($('dl-band'), () => downloadBand(state.info, state.name, pick.cell.name, pick.thumbnail.sample, shownBand()))
})
$('dl-all').addEventListener('click', () => downloading($('dl-all'), async () => {
  const stats = newStats(), stop = followLoad(`All bands · ${pick.cell.name} · for download`, stats)
  try {
    return await downloadAll(state.info, state.name, pick.cell.name, pick.thumbnail.sample, pick.thumbnail.src,
      pick.thumbnail.productId, stats)
  } finally { stop() }
}))
$('dl-python').addEventListener('click', async () => {
  const code = state.info.kind === 'viirs' ? viirsSnippet(state.info, pick.cell.name, pick.thumbnail.sample)
    : pythonSnippet(state.info, state.name, pick.cell.name, pick.thumbnail.sample, pick.bands.length ? pick.bands : ['thumbnail'])
  window.lastSnippet = code                                                         // for automated checks
  try { await navigator.clipboard.writeText(code); $('pick-stats').textContent = 'Python copied to the clipboard' }
  catch { window.prompt('Python that reads this sample:', code) }
})

// ---- the shown image in the full window -----------------------------------------------------------------------
//   opens fitted to the window; scroll zooms about the cursor, drag pans, Esc or Close closes

const view = { scale: 1, x: 0, y: 0, drag: null }

function openImage() {
  if (!pick.shown) return
  const img = $('lightbox-img')
  img.src = pick.shown.src
  $('lightbox-value').textContent = ''
  $('lightbox-title').textContent = `${pick.cell.name} · ${state.info.label} · ${pick.shown.label}` +
    (pick.shown.sample.timestamp ? ` · ${when(pick.shown.sample.timestamp)}` : '')
  $('lightbox').hidden = false
  img.decode().then(() => {
    const w = innerWidth, h = innerHeight - 44, s = Math.min(w / img.naturalWidth, h / img.naturalHeight)
    Object.assign(view, { scale: s, x: (w - img.naturalWidth * s) / 2, y: 44 + (h - img.naturalHeight * s) / 2 })
    place()
  })
}
const place = () => { $('lightbox-img').style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }
const closeImage = () => { $('lightbox').hidden = true }

$('thumb').addEventListener('click', openImage)
$('open-image').addEventListener('click', openImage)
$('lightbox-close').addEventListener('click', closeImage)
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('lightbox').hidden) closeImage() })
$('lightbox').addEventListener('wheel', e => {
  e.preventDefault()
  const k = Math.exp(-e.deltaY * 0.0015), s = Math.min(Math.max(view.scale * k, 0.05), 40)
  view.x = e.clientX - (e.clientX - view.x) * (s / view.scale)
  view.y = e.clientY - (e.clientY - view.y) * (s / view.scale)
  view.scale = s; place()
}, { passive: false })
$('lightbox').addEventListener('pointerdown', e => {
  if (e.target.closest('button')) return
  view.drag = [e.clientX - view.x, e.clientY - view.y]
  $('lightbox').classList.add('dragging'); $('lightbox').setPointerCapture(e.pointerId)
})
$('lightbox').addEventListener('pointermove', e => {
  const v = pick.shown?.values
  if (v) {                                                   // value under the cursor (single bands, elevation)
    const k = v.width / pick.shown.size, x = Math.floor(((e.clientX - view.x) / view.scale) * k)
    const y = Math.floor(((e.clientY - view.y) / view.scale) * k)
    const inside = x >= 0 && y >= 0 && x < v.width && y < v.values.length / v.width
    const value = inside ? v.values[y * v.width + x] : NaN
    $('lightbox-value').textContent = inside && Number.isFinite(value) ? `· pixel ${x}, ${y}: ${+value.toFixed(3)}${v.unit ? ` ${v.unit}` : ''}` : ''
  }
  if (!view.drag) return
  view.x = e.clientX - view.drag[0]; view.y = e.clientY - view.drag[1]; place()
})
$('lightbox').addEventListener('pointerup', () => { view.drag = null; $('lightbox').classList.remove('dragging') })

/** "20200213T023821" -> "2020-02-13 02:38 UTC". */
const when = t => `${t.slice(0, 4)}-${t.slice(4, 6)}-${t.slice(6, 8)} ${t.slice(9, 11)}:${t.slice(11, 13)} UTC`

function meta(sample, productId, info) {
  const rows = []
  if (sample.timestamp) rows.push(['Acquired', when(sample.timestamp)])
  if (sample.cloud_cover !== undefined && sample.cloud_cover !== null) rows.push(['Cloud cover', `${sample.cloud_cover.toFixed(1)}%`])
  if (sample.orbit_state) rows.push(['Orbit', sample.orbit_state])
  if (productId) rows.push(['Product', productId])
  if (info.kind === 'aef') {
    rows.push(['Year', `${sample.year} (annual embedding)${sample.year === sample.cellYear ? ', the Sentinel-2 year' : ''}`])
    rows.push(['Source', `<a href="${info.link}" target="_blank" rel="noopener">tge-labs/aef</a>, CC-BY 4.0`])
  } else if (info.kind === 'viirs') {
    rows.push(['Year', `${sample.year} (annual median)`])
    rows.push(['File', `<a href="${info.files.replace('/resolve/', '/blob/')}${sample.shard}" target="_blank" rel="noopener">${sample.shard.split('/').pop()}</a>, bytes ${sample.offset.toLocaleString('en')} + ${sample.size.toLocaleString('en')}`])
  } else {
    const url = imageUrl(info, sample.part)
    rows.push(['File', `<a href="${url.replace('/resolve/', '/blob/')}" target="_blank" rel="noopener">part_${String(sample.part).padStart(5, '0')}</a>, row ${sample.parquet_row}`])
  }
  $('pick-meta').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')
}
