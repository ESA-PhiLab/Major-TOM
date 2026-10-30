// Place names over the globe from zoom 5: cities and landmarks (places.json, Natural Earth), as HTML labels in the
// page's fonts. After each move: places in view, in the file's order (largest first), each placed unless it would
// overlap one already placed; at most MAX labels. Points behind the globe are left out.

const MIN_ZOOM = 5, MAX = 80, PAD = 3
const MARK = { capital: '■', city: '●', peak: '▲', low: '▼', pass: '⌇', falls: '≈', pole: '+' }

export function placeLabels(map, layer) {
  let rows = [], on = true, frame = 0
  fetch('places.json').then(r => r.json()).then(d => { rows = d.rows; draw() })

  /** Whether a place is on the visible side of the globe (MapLibre's own test where available). */
  const visible = (lon, lat) => {
    const t = map.transform
    if (typeof t?.isLocationOccluded === 'function') return !t.isLocationOccluded({ lng: lon, lat })
    const c = map.getCenter(), rad = Math.PI / 180                                   // within 60° of the centre
    const cos = Math.sin(lat * rad) * Math.sin(c.lat * rad) + Math.cos(lat * rad) * Math.cos(c.lat * rad) * Math.cos((lon - c.lng) * rad)
    return cos > 0.5
  }

  function draw() {
    frame = 0
    layer.replaceChildren()
    if (!on || map.getZoom() < MIN_ZOOM) return
    const { width, height } = map.getCanvas().getBoundingClientRect(), taken = []
    for (const [kind, name, lon, lat, size] of rows) {
      if (taken.length >= MAX) break
      const p = map.project([lon, lat])
      if (p.x < 0 || p.y < 0 || p.x > width || p.y > height || !visible(lon, lat)) continue
      const text = kind === 'peak' || kind === 'low' ? `${name} ${size.toLocaleString('en')} m` : name
      // box estimated from the text (12 px Source Sans: ~6.3 px a character), so only placed labels are made
      const box = { left: p.x - 5, right: p.x + 12 + text.length * 6.3, top: p.y - 9, bottom: p.y + 9 }
      if (taken.some(b => box.left < b.right + PAD && box.right + PAD > b.left && box.top < b.bottom + PAD && box.bottom + PAD > b.top)) continue
      taken.push(box)
      const label = document.createElement('div')
      label.className = `place ${kind}`
      label.innerHTML = `<i>${MARK[kind]}</i><span></span>`
      label.querySelector('span').textContent = text
      label.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)`
      layer.append(label)
    }
  }

  const later = () => { if (!frame) frame = requestAnimationFrame(draw) }
  map.on('move', later)
  map.on('resize', later)
  return { show(value) { on = value; later() } }
}
