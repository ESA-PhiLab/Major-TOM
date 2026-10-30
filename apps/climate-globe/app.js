// The page: a MapLibre globe whose data layer is drawn in the browser, a panel to pick layer, source and month.
//
//   panel (layer, source, month) -> load: preview image or live Parquet (data.js), cached per month
//     -> field (tiles.js) -> new tile URL version -> MapLibre requests tiles -> renderTile -> globe
//   cursor -> cell under it -> outline on the globe + name, position and value in the panel
//   every load -> bytes, requests, read and draw time in the panel (the test this version is for)

import * as maplibregl from 'https://cdn.jsdelivr.net/npm/maplibre-gl@6.14.0/dist/maplibre-gl.mjs'
import { feature } from 'https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/+esm'
import { LAYERS, SOURCE_COOP, cellRow, loadCellClimatology, loadCellSeries, loadPreview, loadSeries } from './data.js'
import { lineChart } from './charts.js'
import { drawBar, makeScale } from './scales.js'
import { cellAt, gridField, imageField, renderTile } from './tiles.js'

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
  'November', 'December']
const $ = id => document.getElementById(id)

const state = { layer: LAYERS[0], mode: 'preview', step: 6, field: null, version: 0, loading: 0 }
const cache = new Map()                                  // "<layer>/<mode>/<step>" -> loaded field, last 8 kept

// ---- map ---------------------------------------------------------------------------------------------------

const EMPTY = { pixel: (lon, lat, out, o) => { out[o + 3] = 0 } }       // before the first layer has loaded
maplibregl.addProtocol('mt', async params => {
  const [z, x, y] = params.url.split('/').slice(-3).map(Number)
  return { data: await renderTile(state.field ?? EMPTY, z, x, y) }
})

const map = new maplibregl.Map({
  container: 'map', center: [12, 22], zoom: innerWidth < 820 ? 1.3 : 2.25, attributionControl: false, maxZoom: 9,
  style: {
    version: 8, projection: { type: 'globe' },
    sky: { 'atmosphere-blend': ['interpolate', ['linear'], ['zoom'], 0, 0.6, 4, 0.3, 7, 0] },
    sources: {
      field: { type: 'raster', tiles: ['mt://0/{z}/{x}/{y}'], tileSize: 256, maxzoom: 7 },
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
map.addControl(new maplibregl.AttributionControl({ compact: true,
  customAttribution: 'Coastlines: Natural Earth · Data: <a href="https://source.coop/major-tom/climate">major-tom/climate</a>' }))

map.on('load', async () => {
  const topo = await (await fetch('https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-50m.json')).json()
  const land = feature(topo, topo.objects.land)
  map.getSource('land').setData(land)
  map.getSource('coast').setData(coastLines(land))
  show()
})

/** Coastlines from the land polygons, without the edges that are not coast: those closing Antarctica along the
 *  date line and the pole, and steps across the date line (Fiji, Chukotka), which a map draws the long way round,
 *  as a line along that latitude across the globe. */
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

let cursor = null                                         // last position under the mouse, on the globe

/** Whether a screen point lies on the globe (not the space around it). MapLibre's own test where available;
 *  otherwise off the globe the map returns a point on the horizon, which projects back somewhere else. */
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

/** Outline and describe the cell under the cursor (again after each load, so the value is never stale). */
function describe() {
  if (!cursor) return
  const { lng, lat } = cursor, cell = cellAt(lng, lat), [w, s, east, n] = cell.bounds
  map.getSource('cell').setData({ type: 'Feature', geometry: { type: 'Polygon',
    coordinates: [[[w, s], [east, s], [east, n], [w, n], [w, s]]] } })
  const v = state.field?.valueAt(lng, lat)
  $('cursor-cell').textContent = cell.name
  $('cursor-pos').textContent = `${lat.toFixed(2)}°, ${lng.toFixed(2)}°`
  $('cursor-value').textContent = v === undefined ? 'colours only (preview)'
    : Number.isNaN(v) ? 'no value' : `${format(v)} ${state.layer.unit}`
}
function clearCursor() {
  cursor = null
  map.getSource('cell').setData({ type: 'FeatureCollection', features: [] })
  for (const id of ['cursor-cell', 'cursor-pos', 'cursor-value']) $(id).textContent = '–'
}
map.getCanvas().addEventListener('mouseleave', clearCursor)

// ---- loading indicator -------------------------------------------------------------------------------------
// The Major TOM Climate flag animates while anything loads: column after column fills top to bottom, then empties
// bottom to top, on a loop. Progress itself is in the text: Loading (MB of the expected total), then Drawing.

const bars = [...document.querySelectorAll('#loader polygon')]                  // 4 columns x 5, each top to bottom
const still = matchMedia('(prefers-reduced-motion: reduce)').matches
let loaderTimer = null, animation = null, tick = 0

function animateFlag() {
  const n = bars.length, k = tick++ % (2 * n)                  // first n ticks fill, next n empty
  if (k < n) bars[k].classList.add('on')
  else { const i = k - n, col = Math.floor(i / 5); bars[col * 5 + 4 - (i % 5)].classList.remove('on') }
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
    if (still) bars.slice(0, 10).forEach(b => b.classList.add('on'))
    else animation = setInterval(animateFlag, 70)
  }, 150)
}
function loaderProgress(share, bytes, expected) {
  $('loader-bytes').textContent = expected ? `${(bytes / 1e6).toFixed(1)} of ${(Math.max(expected, bytes) / 1e6).toFixed(1)} MB`
    : `${(bytes / 1e6).toFixed(1)} MB`
}
function loaderDrawing() {
  $('loader-phase').textContent = 'Drawing'
}
function stopLoader() {
  clearTimeout(loaderTimer); clearInterval(animation)
  $('loader').hidden = true
}

// ---- selected cell ----------------------------------------------------------------------------------------
//   click on the globe -> cell -> outline + "Selected cell" section
//     climatology: one band file, the row groups whose statistics contain the cell -> 12 normals (mean ± std)
//     series (on request): every monthly file, footer + the chunk holding the cell -> chart fills in as they arrive

const INITIALS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
const pick = { cell: null, layer: null, clim: null, series: null, job: null, ticket: 0 }

const outline = ([w, s, e, n]) => ({ type: 'Feature', geometry: { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } })
const kb = bytes => (bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.round(bytes / 1e3)} KB`)
const secs = ms => `${(ms / 1000).toFixed(1)} s`

map.on('click', e => {
  if (!onGlobe(e.point, e.lngLat)) return
  pick.cell = { ...cellAt(e.lngLat.lng, e.lngLat.lat), lng: e.lngLat.lng, lat: e.lngLat.lat }
  map.getSource('selected').setData(outline(pick.cell.bounds))
  $('pick').hidden = false
  $('pick-cell').textContent = pick.cell.name
  $('pick-pos').textContent = `${pick.cell.lat.toFixed(2)}°, ${pick.cell.lng.toFixed(2)}°`
  $('pick').scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  pickLayer()
})
$('pick-clear').addEventListener('click', () => {
  stopSeries(); pick.cell = null; $('pick').hidden = true
  map.getSource('selected').setData({ type: 'FeatureCollection', features: [] })
})

/** The selected cell, for the current layer: its normals now, its series on request. */
async function pickLayer() {
  const layer = state.layer, { row, col } = pick.cell, ticket = ++pick.ticket
  stopSeries()
  pick.layer = layer.id; pick.clim = null; pick.series = null
  $('clim-title').textContent = `${layer.name} (${layer.source.split(',')[0]}), normal ${layer.normal}: mean ± 1 std, ${layer.unit}`
  $('clim-stats').textContent = 'Loading…'
  $('chart-clim').replaceChildren()
  $('series-title').textContent = `Every month, ${layer.series[0]}–${layer.series[1]}`
  $('chart-series').hidden = true
  $('series-stats').textContent = ''
  const months = (layer.series[1] - layer.series[0] + 1) * 12
  $('series-load').textContent = `Load all ${months} months (reads one file per month)`
  $('series-load').hidden = false
  try {
    const c = await loadCellClimatology(layer, row, col)
    if (ticket !== pick.ticket) return
    pick.clim = c
    $('clim-stats').textContent = `${c.stats.rowGroups} row group${c.stats.rowGroups === 1 ? "" : "s"} · ${kb(c.stats.bytes)} in ${c.stats.requests} requests · ${secs(c.stats.read)}`
  } catch (error) {
    if (ticket === pick.ticket) $('clim-stats').textContent = `Could not load: ${error.message}`
  }
  drawPick()
}

function stopSeries() {
  pick.job?.stop(); pick.job = null
}

$('series-load').addEventListener('click', async () => {
  if (pick.job) { stopSeries(); $('series-load').textContent = 'Stopped. Load again'; return }
  const layer = state.layer, ticket = pick.ticket, [a, b] = layer.series, n = (b - a + 1) * 12
  $('series-load').textContent = 'Stop'
  $('series-stats').textContent = 'Finding the cell…'
  const rowIndex = await cellRow(layer.dataset, pick.cell.row, pick.cell.col)
  if (ticket !== pick.ticket) return
  if (rowIndex < 0) { $('series-stats').textContent = `This cell is not in ${layer.dataset}.`; $('series-load').hidden = true; return }
  pick.series = new Array(n).fill(NaN)
  $('chart-series').hidden = false
  let arrived = 0, frame = 0
  const job = loadCellSeries(layer, rowIndex, (k, v, st) => {
    if (ticket !== pick.ticket) return
    pick.series[k] = v; arrived++
    $('series-stats').textContent = `${arrived} of ${n} months · ${kb(st.bytes)} in ${st.requests} requests · ${secs(performance.now() - st.t0)}`
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; drawPick() })
  })
  pick.job = job
  const st = await job.done
  if (ticket !== pick.ticket || pick.job !== job) return
  pick.job = null
  $('series-load').hidden = true
  $('series-stats').textContent = `${n} months · ${kb(st.bytes)} in ${st.requests} requests · ${secs(st.read)} ` +
    `(network ${secs(st.network)})`
  drawPick()
})

/** Draw both charts, the shown month marked. */
function drawPick() {
  if (!pick.cell) return
  const layer = LAYERS.find(l => l.id === pick.layer), { year, month } = when(state.step)
  if (pick.clim) {
    const { mean, std } = pick.clim, floor = layer.scale.type === 'log' ? 0 : -Infinity     // no negative rain
    lineChart($('chart-clim'), {
      x: mean.map((_, i) => i + 1), y: mean, lo: mean.map((m, i) => Math.max(floor, m - std[i])),
      hi: mean.map((m, i) => m + std[i]), xRange: [0.5, 12.5], xTicks: INITIALS.map((t, i) => [i + 1, t]), marker: month - 1 })
  }
  if (pick.series) {
    const [a, b] = layer.series, decades = []
    for (let y = Math.ceil(a / 10) * 10; y <= b; y += 10) decades.push([y, String(y)])
    lineChart($('chart-series'), {
      x: pick.series.map((_, k) => a + (k + 0.5) / 12), y: pick.series, xRange: [a, b + 1], xTicks: decades,
      marker: year === null ? -1 : (year - a) * 12 + month - 1 })
  }
}

// ---- time --------------------------------------------------------------------------------------------------

/** Steps of the time slider: 12 climatology months, or every month of the layer's series. */
function steps() {
  if (state.mode === 'preview') return 12
  const [a, b] = state.layer.series
  return (b - a + 1) * 12
}
function when(step) {                                    // step -> {year, month}; year null for the climatology
  if (state.mode === 'preview') return { year: null, month: step + 1 }
  return { year: state.layer.series[0] + Math.floor(step / 12), month: (step % 12) + 1 }
}
/** {year, month} -> step of the current layer and source; a climatology month goes to the series' last year. */
function stepFor({ year, month }) {
  if (state.mode === 'preview') return month - 1
  const [a, b] = state.layer.series
  return (Math.min(Math.max(year ?? b, a), b) - a) * 12 + month - 1
}
function timeLabel(step) {
  const { year, month } = when(step)
  return year === null ? `${MONTHS[month - 1]} · normal 1991–2020` : `${MONTHS[month - 1]} ${year}`
}

// ---- load and draw -----------------------------------------------------------------------------------------

async function show() {
  const ticket = ++state.loading, { layer, mode, step } = state, key = `${layer.id}/${mode}/${step}`
  $('time-label').textContent = timeLabel(step)
  $('status').textContent = ''
  startLoader(`${layer.name}, ${layer.source.split(',')[0]} · ${timeLabel(step)}`)
  const scale = makeScale(layer.scale)
  let loaded = cache.get(key), cached = Boolean(loaded)
  try {
    if (!loaded) {
      const { year, month } = when(step)
      if (mode === 'preview') {
        const p = await loadPreview(layer, month)
        loaderProgress(1, p.stats.bytes, p.stats.bytes)
        loaded = { field: imageField(p.rgba, p.width, scale), stats: p.stats }
      } else {
        const s = await loadSeries(layer, year, month,
          (share, bytes, expected) => { if (ticket === state.loading) loaderProgress(share, bytes, expected) })
        loaded = { field: gridField(s.values, scale), stats: s.stats }
      }
      cache.set(key, loaded)
      if (cache.size > 8) cache.delete(cache.keys().next().value)
    }
  } catch (error) {
    if (ticket === state.loading) { stopLoader(); $('status').textContent = `Could not load: ${error.message}` }
    return
  }
  if (ticket !== state.loading) return                  // a newer request replaced this one
  state.field = loaded.field
  loaderDrawing()
  describe()
  const t0 = performance.now()
  map.getSource('field').setTiles([`mt://${++state.version}/{z}/{x}/{y}`])
  legend(scale)
  if (pick.cell) { if (pick.layer !== layer.id) pickLayer(); else drawPick() }
  map.once('idle', () => {
    if (ticket !== state.loading) return
    readout(loaded.stats, cached ? null : performance.now() - t0, cached)
    stopLoader()
  })
}

function readout(stats, drawMs, cached) {
  $('stat-source').textContent = state.mode === 'preview' ? 'preview image (PNG)' : 'Parquet, read live'
  $('stat-bytes').textContent = cached ? 'cached' : `${(stats.bytes / 1e6).toFixed(1)} MB in ${stats.requests} request${stats.requests > 1 ? 's' : ''}` +
    (stats.firstOfDataset ? ' (with cell positions)' : '')
  $('stat-read').textContent = cached ? '–' : stats.network === undefined ? `${Math.round(stats.read)} ms`
    : `${Math.round(stats.read)} ms (network ${Math.round(stats.network)})`
  $('stat-draw').textContent = drawMs === null ? '–' : `${Math.round(drawMs)} ms`
  $('stat-cells').textContent = stats.cells ? stats.cells.toLocaleString('en') : '–'
  document.body.dataset.loaded = `${state.layer.id}/${state.mode}/${state.step}`     // for automated checks
}

function legend(scale) {
  drawBar($('legend-bar'), scale)
  $('legend-ticks').replaceChildren(...scale.ticks.map(t => {
    const span = document.createElement('span')
    span.textContent = scale.type === 'centred' && t > 0 ? `+${t}` : `${t}`
    span.style.left = `${scale.position(t) * 100}%`
    return span
  }))
  $('legend-unit').textContent = `${state.layer.unit}${scale.type === 'log' ? ' · log scale' : ''}`
}

const format = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toPrecision(2))

// ---- panel -------------------------------------------------------------------------------------------------

function buildLayerList() {
  $('layers').replaceChildren(...LAYERS.map(layer => {
    const label = document.createElement('label')
    label.className = 'option'
    label.innerHTML = `<input type="radio" name="layer" value="${layer.id}"><span><b>${layer.name}</b>` +
      `<small>${layer.source}</small></span>`
    label.querySelector('input').checked = layer === state.layer
    label.querySelector('input').addEventListener('change', () => {
      const t = when(state.step)                       // keep the month (and year, where the new layer has it)
      state.layer = layer; syncControls(); state.step = stepFor(t); syncControls(); show()
    })
    return label
  }))
}

/** Source and time controls follow the layer: a layer without preview images is read live. */
function syncControls() {
  const hasPreview = Boolean(state.layer.preview)
  $('mode-preview').disabled = !hasPreview
  if (!hasPreview) state.mode = 'live'
  for (const b of document.querySelectorAll('[data-mode]')) b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode))
  $('mode-note').textContent = hasPreview ? '' : 'No preview images for this layer yet: read live.'
  const n = steps()
  state.step = Math.min(state.step, n - 1)
  $('time').max = String(n - 1); $('time').value = String(state.step)
  $('time-range').textContent = state.mode === 'preview' ? 'Jan – Dec' : `${state.layer.series[0]} – ${state.layer.series[1]}`
  $('dataset-link').href = `https://source.coop/major-tom/climate/${state.layer.dataset}`
  $('dataset-link').textContent = state.layer.dataset
}

function move(delta) {
  const n = steps()
  state.step = (state.step + delta + n) % n
  $('time').value = String(state.step)
  show()
}

for (const b of document.querySelectorAll('[data-mode]')) {
  b.addEventListener('click', () => {
    if (b.disabled) return
    const t = when(state.step)
    state.mode = b.dataset.mode; state.step = stepFor(t); syncControls(); show()
  })
}
$('time').addEventListener('input', e => { state.step = Number(e.target.value); $('time-label').textContent = timeLabel(state.step) })
$('time').addEventListener('change', () => show())
$('prev').addEventListener('click', () => move(-1))
$('next').addEventListener('click', () => move(1))
// ← → change the month and nothing else: handled before the map (which would pan) and the layer buttons (which
// would switch layer) see them. With Shift, Alt, Ctrl or Meta they stay the map's (rotation).
document.addEventListener('keydown', e => {
  if (e.target.closest('input[type=range]') || !['ArrowLeft', 'ArrowRight'].includes(e.key)) return
  if (e.shiftKey || e.altKey || e.ctrlKey || e.metaKey) return
  e.preventDefault(); e.stopPropagation()
  move(e.key === 'ArrowLeft' ? -1 : 1)
}, { capture: true })
$('source-base').textContent = SOURCE_COOP.replace('https://', '')

buildLayerList()
syncControls()
