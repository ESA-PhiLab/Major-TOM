// The page: a MapLibre globe of where a Core dataset has samples, and the image of the sample in a clicked cell.
//
//   dataset (panel) -> footprints/<name>.parquet, two position columns (footprints.js) -> mask on the cell array
//     -> coverage tiles drawn in the browser (tiles.js) -> globe
//   click -> cell -> its samples (footprint rows, page index) -> image file on Hugging Face, one row group,
//     thumbnail column (samples.js) -> panel
//   cursor -> cell under it -> outline + name, position, whether it has a sample
//   the shown image -> full window (zoom, pan) or on the map over its footprint (corners from its UTM centre)

import * as maplibregl from 'https://cdn.jsdelivr.net/npm/maplibre-gl@6.14.0/dist/maplibre-gl.mjs'
import { feature } from 'https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/+esm'
import proj4 from 'https://cdn.jsdelivr.net/npm/proj4@2.12.1/+esm'
import { DATASETS, loadCoverage, samplesAt } from './footprints.js'
import { placeLabels } from './labels.js'
import { imageUrl, loadThumbnail } from './samples.js'
import { cellAt, cellIndex, renderTile } from './tiles.js'

const $ = id => document.getElementById(id)
const COLOURS = { S2L2A: [57, 181, 74], S2L1C: [0, 113, 188], S1RTC: [241, 90, 36], DEM: [180, 140, 90] }

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
  container: 'map', center: [12, 22], zoom: innerWidth < 820 ? 1.3 : 2.25, attributionControl: false, maxZoom: 12,
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
  $('dataset-link').href = `https://huggingface.co/datasets/Major-TOM/Core-${name}`
  $('dataset-link').textContent = `huggingface.co/datasets/Major-TOM/Core-${name}`
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
    $('stat-footprints').textContent = `${kb(coverage.stats.bytes)} in ${coverage.stats.requests} requests · ${secs(coverage.stats.read)}`
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
  if (!lookups.has(key)) lookups.set(key, samplesAt(name, row, col).catch(e => { lookups.delete(key); throw e }))
  return lookups.get(key)
}
function thumbnailOf(name, sample) {
  const key = `${name}/${sample.part}/${sample.parquet_row}`, fresh = !thumbs.has(key)
  if (fresh) thumbs.set(key, loadThumbnail(state.datasets[name], sample).catch(e => { thumbs.delete(key); throw e }))
  return [thumbs.get(key), fresh]
}

/** The selected cell in every other dataset, in the panel's order, after the shown one. */
async function prefetchCell(cell) {
  for (const name of Object.keys(state.datasets)) {
    if (pick.cell !== cell) return                         // another cell was picked
    if (name === state.name) continue
    try {
      const samples = await samplesOf(name, cell.row, cell.col)
      if (samples.length) await thumbnailOf(name, samples[0])[0]
    } catch { /* loaded again, with its error shown, when that dataset is shown */ }
  }
}

const pick = { cell: null, samples: [], chosen: 0, thumbnail: null, ticket: 0 }

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
  pick.samples = []; pick.thumbnail = null
  showThumbnail(null, 'Looking up the cell…'); showOnMap()
  $('choices').replaceChildren(); $('pick-meta').replaceChildren(); $('pick-stats').textContent = ''
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
      b.textContent = s.timestamp ? when(s.timestamp) : `sample ${i + 1}`
      b.addEventListener('click', () => choose(i))
      return b
    }))
  }
  choose(0)
}

async function choose(i) {
  const ticket = ++pick.ticket, sample = pick.samples[i], info = state.info
  pick.chosen = i
  for (const [k, b] of [...$('choices').children].entries()) b.setAttribute('aria-pressed', String(k === i))
  meta(sample, null, info)
  showThumbnail(null, 'Loading the thumbnail…')
  try {
    const [loading, fresh] = thumbnailOf(state.name, sample), t = await loading
    if (ticket !== pick.ticket) return
    pick.thumbnail = { ...t, sample }
    showThumbnail(t.src)
    showOnMap()
    meta(sample, t.productId, info)
    $('pick-stats').textContent = `Thumbnail: ${kb(t.stats.bytes)} in ${t.stats.requests} requests · ${secs(t.stats.read)}` +
      (fresh ? '' : ' (loaded before)')
    prefetchCell(pick.cell)
  } catch (error) {
    if (ticket === pick.ticket) showThumbnail(null, `Could not load: ${error.message}`)
  }
}

function showThumbnail(src, note = '') {
  $('thumb').hidden = !src
  $('image-actions').hidden = !src
  if (src) $('thumb').src = src
  $('thumb-note').textContent = src ? '' : note
}

// ---- the shown image on the map -------------------------------------------------------------------------------
// Every Core image is 10,680 m square in its UTM zone, centred on the sample's centre_lat, centre_lon (checked
// against the GeoTIFF bounds), north up: its corners are the centre +- 5,340 m, back in longitude and latitude.

const HALF = 5340

/** Corners of a sample's image, [lon, lat], from top left clockwise (as MapLibre's image source wants them). */
function corners(s) {
  const zone = s.epsg % 100, south = s.epsg > 32700
  const utm = proj4(`+proj=utm +zone=${zone}${south ? ' +south' : ''} +datum=WGS84 +units=m +no_defs`)
  const [x, y] = utm.forward([s.centre_lon, s.centre_lat])
  return [[x - HALF, y + HALF], [x + HALF, y + HALF], [x + HALF, y - HALF], [x - HALF, y - HALF]].map(p => utm.inverse(p))
}

/** The shown thumbnail over its footprint when "Show on map" is ticked; otherwise none. */
function showOnMap() {
  const t = pick.thumbnail, on = $('on-map').checked && t && pick.cell
  if (map.getLayer('sample')) map.removeLayer('sample')
  if (map.getSource('sample')) map.removeSource('sample')
  if (!on) return
  map.addSource('sample', { type: 'image', url: t.src, coordinates: corners(t.sample) })
  map.addLayer({ id: 'sample', type: 'raster', source: 'sample',
    paint: { 'raster-opacity': Number($('opacity').value) / 100, 'raster-fade-duration': 0 } }, 'coast')
}
$('on-map').addEventListener('change', showOnMap)
$('opacity').addEventListener('input', () => {
  if (map.getLayer('sample')) map.setPaintProperty('sample', 'raster-opacity', Number($('opacity').value) / 100)
})
window.sampleCorners = () => (pick.thumbnail ? corners(pick.thumbnail.sample) : null)     // for automated checks

// ---- the shown image in the full window -----------------------------------------------------------------------
//   opens fitted to the window; scroll zooms about the cursor, drag pans, Esc or Close closes

const view = { scale: 1, x: 0, y: 0, drag: null }

function openImage() {
  if (!pick.thumbnail) return
  const img = $('lightbox-img')
  img.src = pick.thumbnail.src
  $('lightbox-title').textContent = `${pick.cell.name} · ${state.info.label}` +
    (pick.thumbnail.sample.timestamp ? ` · ${when(pick.thumbnail.sample.timestamp)}` : '')
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
  const url = imageUrl(info, sample.part)
  rows.push(['File', `<a href="${url.replace('/resolve/', '/blob/')}" target="_blank" rel="noopener">part_${String(sample.part).padStart(5, '0')}</a>, row ${sample.parquet_row}`])
  $('pick-meta').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')
}
