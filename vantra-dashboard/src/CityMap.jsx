import { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet.heat'
import 'leaflet/dist/leaflet.css'

const BLR_CENTER = [12.955, 77.61]
const PLATE_COLORS = ['#58a6ff', '#3fb950', '#bc8cff', '#f85149', '#d29922', '#39c5cf']

export default function CityMap({
  cameras, edges, trajectory, heatmap, selectedDet, onSelectDet, focusOn,
  liveEvents, compareTrajectories, whatifMode, whatifBusy, whatifPoint, whatifResult, onMapClick,
  activeSession,
}) {
  const el = useRef(null)
  const map = useRef(null)
  const layers = useRef({ cams: null, edges: null, traj: null, heat: null, live: null, compare: null, hypo: null })
  const [pop, setPop] = useState(null)

  useEffect(() => {
    if (map.current) return
    map.current = L.map(el.current, { zoomControl: true }).setView(BLR_CENTER, 13)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap', maxZoom: 19,
    }).addTo(map.current)
    layers.current.cams = L.layerGroup().addTo(map.current)
    layers.current.edges = L.layerGroup().addTo(map.current)
    layers.current.traj = L.layerGroup().addTo(map.current)
    layers.current.heat = L.layerGroup().addTo(map.current)
    layers.current.live = L.layerGroup().addTo(map.current)
    layers.current.compare = L.layerGroup().addTo(map.current)
    layers.current.hypo = L.layerGroup().addTo(map.current)
    map.current.on('click', (e) => {
      setPop(null)
      if (onMapClick) onMapClick(e.latlng.lat, e.latlng.lng)
    })
  }, [])

  // keep click handler fresh
  useEffect(() => {
    if (!map.current) return
    map.current.off('click')
    map.current.on('click', (e) => {
      setPop(null)
      if (onMapClick) onMapClick(e.latlng.lat, e.latlng.lng)
    })
  }, [onMapClick])

  useEffect(() => {
    const g = layers.current.cams
    if (!g || !cameras) return
    g.clearLayers()
    cameras.forEach(c => {
      L.circleMarker([c.lat, c.lng], {
        radius: 6, color: '#58a6ff', fillColor: '#0d1117', weight: 2, fillOpacity: 1,
      })
        .bindPopup(`<b>${c.camera_id}</b><br>${c.name}<br><span style="color:#8b949e">${c.road || ''}</span>`)
        .addTo(g)
    })
  }, [cameras])

  useEffect(() => {
    const g = layers.current.edges
    if (!g || !edges) return
    g.clearLayers()
    edges.forEach(e => {
      const a = cameras.find(c => c.camera_id === e.src_cam)
      const b = cameras.find(c => c.camera_id === e.dst_cam)
      if (!a || !b) return
      L.polyline([[a.lat, a.lng], [b.lat, b.lng]], {
        color: '#30363d', weight: 1.5, dashArray: '2,4', interactive: false,
      }).addTo(g)
    })
  }, [edges, cameras])

  useEffect(() => {
    const g = layers.current.heat
    if (!g) return
    g.clearLayers()
    if (!heatmap) return
    const max = Math.max(...heatmap.map(h => h.weight), 1)
    const pts = heatmap.map(h => [h.lat, h.lng, h.weight / max])
    if (pts.length) {
      L.heatLayer(pts, {
        radius: 40, blur: 18, maxZoom: 13, minOpacity: 0.55,
        gradient: { 0.2: '#1f6feb', 0.45: '#58a6ff', 0.65: '#d29922', 0.85: '#f85149', 1.0: '#ff7b72' },
      }).addTo(g)
    }
  }, [heatmap])

  useEffect(() => {
    const g = layers.current.traj
    if (!g) return
    g.clearLayers()
    if (!trajectory) return
    const pts = trajectory.detections.map(d => [d.lat, d.lng])
    if (pts.length > 1) {
      for (let i = 0; i < pts.length - 1; i++) {
        const hop = trajectory.hops.find(h => h.seq === i)
        const tier = !hop ? 'unknown'
          : hop.anomaly ? 'anomaly'
          : hop.confidence > 0.8 ? 'confirmed'
          : hop.confidence > 0.55 ? 'uncertain' : 'anomaly'
        const style = {
          unknown:   { color: '#8b949e', weight: 4, opacity: 0.9 },
          confirmed: { color: '#3fb950', weight: 4, opacity: 0.9 },
          uncertain: { color: '#ff8c00', weight: 5, opacity: 1 },
          anomaly:   { color: '#f85149', weight: 5, opacity: 1 },
        }[tier]
        // white casing under uncertain/anomaly hops so they pop on the bright map
        if (tier === 'uncertain' || tier === 'anomaly') {
          L.polyline([pts[i], pts[i + 1]], {
            color: '#ffffff', weight: 8, opacity: 0.85, interactive: false,
          }).addTo(g)
        }
        L.polyline([pts[i], pts[i + 1]], { ...style }).addTo(g)
      }
    }
    const demoMode = !!activeSession
    trajectory.detections.forEach((d, i) => {
      const m = L.circleMarker([d.lat, d.lng], {
        radius: 11, color: '#e6edf3', weight: 2,
        fillColor: demoMode ? '#3fb950' : '#1f6feb', fillOpacity: 1,
      }).addTo(g)
      m.bindTooltip(`${i + 1}`, { permanent: true, direction: 'center', className: 'node-label' })
      m.on('click', (ev) => {
        L.DomEvent.stopPropagation(ev)
        const hop = trajectory.hops.find(h => h.to_det === d.det_id)
        const rect = el.current.getBoundingClientRect()
        setPop({ x: ev.containerPoint.x + 12, y: ev.containerPoint.y + 12, det: d, hop, idx: i, rect })
        onSelectDet?.(d.det_id)
      })
    })
    if (pts.length) map.current.fitBounds(L.latLngBounds(pts).pad(0.25))
  }, [trajectory, activeSession])

  // ---------- live replay events ----------
  useEffect(() => {
    const g = layers.current.live
    if (!g) return
    g.clearLayers()
    if (!liveEvents || !liveEvents.length) return
    liveEvents.forEach(ev => {
      const m = L.circleMarker([ev.lat, ev.lng], {
        radius: ev.is_new ? 8 : 5,
        color: ev.is_new ? '#3fb950' : '#1f6feb',
        fillColor: '#3fb950',
        fillOpacity: ev.is_new ? 1 : 0.25,
        weight: 2,
      }).addTo(g)
      m.bindPopup(`<b>${ev.camera_id}</b> ${ev.plate || '(unreadable)'}<br>${ev.ts.slice(11, 19)}`)
    })
  }, [liveEvents])

  // ---------- compare view ----------
  useEffect(() => {
    const g = layers.current.compare
    if (!g) return
    g.clearLayers()
    if (!compareTrajectories || !compareTrajectories.length) return
    const allPts = []
    compareTrajectories.forEach((ct, idx) => {
      const color = PLATE_COLORS[idx % PLATE_COLORS.length]
      const pts = ct.detections.map(d => [d.lat, d.lng])
      allPts.push(...pts)
      if (pts.length > 1) {
        L.polyline(pts, { color, weight: 4, opacity: 0.9 }).addTo(g)
      }
      ct.detections.forEach(d => {
        const m = L.circleMarker([d.lat, d.lng], {
          radius: 8, color: '#e6edf3', weight: 2, fillColor: color, fillOpacity: 1,
        }).addTo(g)
        m.bindTooltip(`${ct.plate} @ ${d.camera_id}`, { direction: 'top' })
      })
    })
    if (allPts.length) map.current.fitBounds(L.latLngBounds(allPts).pad(0.25))
  }, [compareTrajectories])

  // ---------- what-if mode: crosshair cursor ----------
  useEffect(() => {
    if (!el.current) return
    el.current.style.cursor = whatifMode ? 'crosshair' : ''
  }, [whatifMode])

  // ---------- what-if: marker at the selected spot + connection lines ----------
  useEffect(() => {
    const g = layers.current.hypo
    if (!g) return
    g.clearLayers()
    if (!whatifPoint) return
    const { lat, lng } = whatifPoint
    // connection lines to the cameras the hypo camera would link to
    const connected = whatifResult?.hypothetical_camera?.connects_to || []
    connected.forEach(cid => {
      const c = cameras.find(x => x.camera_id === cid)
      if (!c) return
      L.polyline([[lat, lng], [c.lat, c.lng]], {
        color: '#bc8cff', weight: 2, dashArray: '6,6', opacity: 0.85, interactive: false,
      }).addTo(g)
    })
    // pulse ring + marker; gray pulsing ring while the simulation runs
    const busyColor = whatifBusy ? '#8b949e' : '#bc8cff'
    const ring = L.circleMarker([lat, lng], {
      radius: 18, color: busyColor, weight: 2, fillOpacity: 0.15, interactive: false,
    }).addTo(g)
    if (whatifBusy) {
      let r = 18
      const pulse = setInterval(() => {
        r = r >= 34 ? 18 : r + 2
        ring.setRadius(r)
      }, 120)
      // stop the pulse when the result arrives (effect re-runs on whatifResult change)
      setTimeout(() => clearInterval(pulse), 30000)
    }
    const m = L.circleMarker([lat, lng], {
      radius: 9, color: '#e6edf3', weight: 2, fillColor: '#bc8cff', fillOpacity: 1,
    }).addTo(g)
    m.bindTooltip('proposed camera', { direction: 'top', permanent: false })
    if (connected.length) {
      map.current.fitBounds(L.latLngBounds(
        [[lat, lng], ...connected.map(cid => {
          const c = cameras.find(x => x.camera_id === cid)
          return c ? [c.lat, c.lng] : null
        }).filter(Boolean)]).pad(0.3))
    } else {
      map.current.setView([lat, lng], 14)
    }
  }, [whatifPoint, whatifResult, cameras])

  useEffect(() => {
    if (focusOn && cameras.length) {
      const c = cameras.find(x => x.camera_id === focusOn)
      if (c) map.current.setView([c.lat, c.lng], 14)
    }
  }, [focusOn, cameras])

  return (
    <div className="map-pane" ref={el} style={{ height: '100%' }}>
      {pop && (
        <div className="evidence-pop" style={{
          left: Math.min(pop.x, (pop.rect?.width || 800) - 350),
          top: Math.min(pop.y, (pop.rect?.height || 600) - 320),
        }} onClick={e => e.stopPropagation()}>
          <span className="close" onClick={() => setPop(null)}>✕</span>
          <div>
            <b>{pop.idx + 1}. {pop.det.camera_id}</b>
            <span style={{ color: 'var(--muted)' }}> · {new Date(pop.det.ts).toLocaleString()}</span>
          </div>
          <div style={{ fontFamily: 'ui-monospace, monospace', margin: '4px 0' }}>
            plate read: <b>{pop.det.plate_raw || '— unreadable —'}</b>{' '}
            (conf {pop.det.ocr_conf ? (pop.det.ocr_conf * 100).toFixed(0) + '%' : 'n/a'})
          </div>
          {pop.det.crop_url
            ? <img src={pop.det.crop_url} alt="vehicle crop" />
            : <div style={{ color: 'var(--muted)', fontSize: 11 }}>no stored crop for this detection</div>}
          {pop.det.plate_crop_url && <img src={pop.det.plate_crop_url} alt="plate crop" style={{ width: 180 }} />}
          <div>{pop.det.vehicle_type} · {pop.det.vehicle_color}</div>
          {pop.hop && (
            <>
              <hr style={{ borderColor: 'var(--border)', margin: '8px 0' }} />
              <div>
                <span className={`tier tier-${pop.hop.tier}`}>{pop.hop.tier}</span>
                linked to previous node at {(pop.hop.confidence * 100).toFixed(0)}% confidence
              </div>
              <div className="reason" style={{ color: 'var(--muted)', margin: '4px 0' }}>{pop.hop.reason}</div>
              <table>
                <tbody>
                  {pop.hop.edit_distance != null && <tr><td>edit distance</td><td>{pop.hop.edit_distance}</td></tr>}
                  {pop.hop.appearance_score != null && <tr><td>re-id cosine</td><td>{pop.hop.appearance_score.toFixed(2)}</td></tr>}
                  {pop.hop.time_score != null && <tr><td>time score</td><td>{pop.hop.time_score.toFixed(2)}</td></tr>}
                  {pop.hop.ocr_conf_min != null && <tr><td>ocr conf (min)</td><td>{pop.hop.ocr_conf_min.toFixed(2)}</td></tr>}
                  {pop.hop.hop_reach === 2 && <tr><td>hop reach</td><td>2 (one camera missed)</td></tr>}
                  {pop.hop.anomaly && <tr><td>anomaly</td><td style={{ color: 'var(--red)' }}>{pop.hop.anomaly}</td></tr>}
                </tbody>
              </table>
            </>
          )}
          {!pop.hop && <div style={{ color: 'var(--muted)', marginTop: 6 }}>first node — no incoming hop</div>}
        </div>
      )}
    </div>
  )
}
