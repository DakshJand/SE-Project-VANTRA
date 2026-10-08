import { useEffect, useState } from 'react'
import CityMap from './CityMap.jsx'
import LivePanel from './LivePanel.jsx'
import { ComparePanel, WhatIfPanel, TuningPanel, AuditPanel, NotifyPanel, CMP_COLORS } from './FeaturePanels.jsx'

const API = ''
const OPERATORS = ['op-alice', 'op-bob', 'analyst-carol']

export default function App() {
  const [cameras, setCameras] = useState([])
  const [edges, setEdges] = useState([])
  const [heatmap, setHeatmap] = useState(null)
  const [showHeat, setShowHeat] = useState(false)
  const [query, setQuery] = useState('')
  const [searchRes, setSearchRes] = useState(null)
  const [trajectory, setTrajectory] = useState(null)
  const [activeSession, setActiveSession] = useState(null)
  const [tab, setTab] = useState('trajectory')
  const [alerts, setAlerts] = useState([])
  const [scenarios, setScenarios] = useState([])
  const [selectedDet, setSelectedDet] = useState(null)
  const [focusCam, setFocusCam] = useState(null)
  const [status, setStatus] = useState('loading…')
  // feature-expansion state
  const [role, setRole] = useState('operator')           // operator | analyst
  const [operator, setOperator] = useState('op-alice')
  const [liveEvents, setLiveEvents] = useState([])
  const [comparePlates, setComparePlates] = useState([])
  const [compareResult, setCompareResult] = useState(null)
  const [whatifCandidates, setWhatifCandidates] = useState([])
  const [whatifResult, setWhatifResult] = useState(null)
  const [whatifBusy, setWhatifBusy] = useState(false)
  const [whatifMode, setWhatifMode] = useState(false)
  const [whatifPoint, setWhatifPoint] = useState(null)
  const [isMobile, setIsMobile] = useState(window.innerWidth < 760)

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 760)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  // send operator identity on every API call via fetch wrapper
  const api = (path, opts = {}) => {
    const headers = { ...(opts.headers || {}), 'x-vantra-operator': operator }
    return fetch(path, { ...opts, headers })
  }
  window._vantraApi = api   // used by LivePanel etc.

  useEffect(() => {
    ;(async () => {
      const [c, e, hm, al, sc, wc] = await Promise.all([
        fetch(`${API}/api/cameras`).then(r => r.json()),
        fetch(`${API}/api/edges`).then(r => r.json()),
        fetch(`${API}/api/analytics/heatmap?hours_back=1440`).then(r => r.json()),
        fetch(`${API}/api/alerts?limit=50`).then(r => r.json()),
        fetch(`${API}/api/demo/scenarios`).then(r => r.json()),
        fetch(`${API}/api/whatif/candidates`).then(r => r.json()).catch(() => []),
      ])
      setCameras(c); setEdges(e); setHeatmap(hm); setAlerts(al); setScenarios(sc)
      setWhatifCandidates(wc || [])
      setStatus(`${c.length} cameras · ${e.length} edges`)
    })()
  }, [])

  const doSearch = async () => {
    if (query.trim().length < 2) return
    const r = await fetch(`${API}/api/search?q=${encodeURIComponent(query.trim())}`).then(r => r.json())
    setSearchRes(r)
  }

  const pickPlate = async (plate) => {
    setSearchRes(null)
    setActiveSession(null)
    setCompareResult(null)
    const r = await api(`${API}/api/trajectories/reconstruct?plate=${plate}`).then(r => r.json())
    if (Array.isArray(r) && r.length) {
      setTrajectory(r.sort((a, b) => b.n_detections - a.n_detections)[0])
      setTab('trajectory')
    } else {
      setTrajectory(null)
    }
  }

  const runScenario = async (sc) => {
    setActiveSession(sc.key)
    const p = sc.payload || {}
    const r = await api(
      `${API}/api/trajectories/reconstruct?plate=${sc.plate}&session=${p.session}`
    ).then(r => r.json())
    if (Array.isArray(r) && r.length) {
      setTrajectory(r[0])
      setTab('trajectory')
      const al = await fetch(`${API}/api/alerts?limit=50`).then(r => r.json())
      setAlerts(al)
    }
    setFocusCam(p.route ? p.route[0] : null)
  }

  const exportPdf = () => {
    if (!trajectory) return
    const s = activeSession
      ? (scenarios.find(x => x.key === activeSession)?.payload?.session)
      : null
    const url = s
      ? `${API}/api/trajectories/reconstruct/pdf?plate=${trajectory.plate}&session=${s}`
      : `${API}/api/trajectories/reconstruct/pdf?plate=${trajectory.plate}`
    window.open(url, '_blank')
  }

  const runCompare = async () => {
    const r = await api(`${API}/api/compare`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ plates: comparePlates }),
    }).then(r => r.json())
    setCompareResult(r)
    setTrajectory(null)
    setLiveEvents([])
  }

  const evaluateWhatIf = async (lat, lng) => {
    setWhatifBusy(true); setWhatifResult(null); setWhatifPoint({ lat, lng })
    const r = await api(`${API}/api/whatif/evaluate?lat=${lat}&lng=${lng}`, {
      method: 'POST',
    }).then(r => r.json())
    setWhatifResult(r); setWhatifBusy(false)
  }

  const onLiveEvents = (events) => {
    setTrajectory(null)
    setCompareResult(null)
    setLiveEvents(prev => [...prev, ...events].slice(-80))
  }

  const confClass = c => c > 0.8 ? 'conf-high' : c > 0.5 ? 'conf-mid' : 'conf-low'

  const TABS = isMobile
    ? ['search', 'alerts']
    : role === 'operator'
      ? ['live', 'trajectory', 'alerts', 'compare', 'demo', 'audit', 'tuning', 'notify']
      : ['live', 'trajectory', 'analytics', 'compare', 'whatif', 'demo', 'audit', 'tuning']

  // ------------------------------------------------ mobile read-only view
  if (isMobile) {
    return (
      <div className="app mobile">
        <header className="header">
          <span className="logo">VANTRA</span>
          <span className="spacer" />
          <span style={{ color: 'var(--muted)', fontSize: 11 }}>{status}</span>
        </header>
        <div style={{ padding: 10 }}>
          <div className="searchbox" style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
            <input style={{ flex: 1 }} placeholder="plate lookup…"
                   value={query} onChange={e => setQuery(e.target.value.toUpperCase())}
                   onKeyDown={e => e.key === 'Enter' && doSearch()} />
            <button onClick={doSearch}>Go</button>
          </div>
          {searchRes && searchRes.matches.slice(0, 10).map(m => (
            <button className="scenario" key={m.plate} onClick={() => pickPlate(m.plate)}>
              <b style={{ fontFamily: 'ui-monospace, monospace' }}>{m.plate}</b>
              <span style={{ color: 'var(--muted)', fontSize: 11 }}> · {m.n_detections} dets</span>
            </button>
          ))}
          {trajectory && (
            <div className="card">
              <h3>{trajectory.plate}</h3>
              <span className={`conf-badge ${confClass(trajectory.confidence)}`}>
                {Math.round(trajectory.confidence * 100)}%
              </span>{' '}
              <span style={{ color: 'var(--muted)', fontSize: 12 }}>
                {trajectory.n_detections} dets · {trajectory.hops.length} hops
              </span>
              {trajectory.detections.map((d, i) => (
                <div key={d.det_id} style={{ fontSize: 12, marginTop: 6 }}>
                  {i + 1}. <b>{d.camera_id}</b> {new Date(d.ts).toLocaleTimeString()}{' '}
                  <span style={{ fontFamily: 'ui-monospace, monospace' }}>{d.plate_raw || '—'}</span>
                </div>
              ))}
            </div>
          )}
          <div className="card">
            <h3>Alerts</h3>
            {alerts.slice(0, 20).map(a => (
              <div key={a.alert_id} className="alert" style={{ fontSize: 12 }}>
                <b>{a.kind.replace('_', ' ')}</b> · {a.title}
              </div>
            ))}
            {!alerts.length && <div className="loading">no alerts</div>}
          </div>
        </div>
      </div>
    )
  }

  // ------------------------------------------------ desktop
  return (
    <div className="app">
      <header className="header">
        <span className="logo">VANTRA</span>
        <span className="tagline">One vehicle, one trail, across every camera in the city.</span>
        <span className="spacer" />
        <div className="searchbox">
          <input
            placeholder="plate or wildcard e.g. KA01* / KA?1AB…"
            value={query}
            onChange={e => setQuery(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && doSearch()}
          />
          <button onClick={doSearch}>Search</button>
        </div>
        <button onClick={() => { setShowHeat(!showHeat); if (!showHeat) setRole('analyst') }}>
          {showHeat ? 'Hide' : 'Show'} heatmap
        </button>
        <select value={operator} onChange={e => setOperator(e.target.value)}
                title="mock operator identity (audit logging)"
                style={{ background: 'var(--bg)', color: 'var(--text)',
                         border: '1px solid var(--border)', borderRadius: 6, padding: 6, fontSize: 12 }}>
          {OPERATORS.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <div className="role-toggle">
          <button className={role === 'operator' ? 'active' : ''} onClick={() => {
            setRole('operator'); setShowHeat(false)
            setWhatifMode(false); setWhatifPoint(null); setWhatifResult(null)
          }}>Operator</button>
          <button className={role === 'analyst' ? 'active' : ''} onClick={() => setRole('analyst')}>Analyst</button>
        </div>
        <span style={{ color: 'var(--muted)', fontSize: 12 }}>{status}</span>
      </header>

      <div className="main">
        <div style={{ position: 'relative', overflow: 'hidden' }}>
          <CityMap
            cameras={cameras}
            edges={edges}
            trajectory={trajectory}
            heatmap={showHeat ? heatmap : null}
            selectedDet={selectedDet}
            onSelectDet={setSelectedDet}
            focusOn={focusCam}
            liveEvents={liveEvents.length ? liveEvents : null}
            compareTrajectories={compareResult?.trajectories || null}
            activeSession={activeSession}
            whatifMode={whatifMode}
            whatifBusy={whatifBusy}
            whatifPoint={whatifPoint}
            whatifResult={whatifResult}
            onMapClick={whatifMode ? ((lat, lng) => evaluateWhatIf(lat, lng)) : null}
          />
          {whatifMode && (
            <div className="whatif-banner">
              📍 Click the map to place a hypothetical camera
              <button className="btn" style={{ marginLeft: 10 }} onClick={() => {
                setWhatifMode(false); setWhatifPoint(null); setWhatifResult(null)
              }}>Exit</button>
            </div>
          )}
        </div>

        <aside className="side">
          {searchRes && (
            <div className="card">
              <h3>Search — {searchRes.matches.length} match(es)</h3>
              <div className="search-results">
                {searchRes.matches.map(m => (
                  <div className="row" key={m.plate} onClick={() => pickPlate(m.plate)}>
                    <span>{m.plate}</span>
                    <span style={{ color: 'var(--muted)' }}>{m.n_detections} dets</span>
                  </div>
                ))}
                {!searchRes.matches.length && <div className="loading">no matches</div>}
              </div>
            </div>
          )}

          <div className="tabs">
            {TABS.map(t => (
              <button key={t} className={tab === t ? 'active' : ''} onClick={() => {
                setTab(t)
                if (t === 'whatif') setWhatifMode(true)
                else {
                  setWhatifMode(false)
                  setWhatifPoint(null)
                  setWhatifResult(null)
                }
              }}>
                {t === 'trajectory' ? 'Trajectory' : t[0].toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {tab === 'live' && (
            <LivePanel onEvents={onLiveEvents} onAlerts={(a) => setAlerts(prev => {
              const titles = new Set(prev.map(x => x.title))
              return [...prev, ...a.filter(x => !titles.has(x.title))]
            })} />
          )}

          {tab === 'trajectory' && (
            <>
              {trajectory ? (
                <>
                  <div className="card">
                    <h3>Trajectory — {trajectory.plate}</h3>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                      <span className={`conf-badge ${confClass(trajectory.confidence)}`}>
                        {Math.round(trajectory.confidence * 100)}% overall
                      </span>
                      <span style={{ color: 'var(--muted)', fontSize: 12 }}>
                        {trajectory.n_detections} detections · {trajectory.hops.length} hops
                      </span>
                      <span style={{ flex: 1 }} />
                      <button className="btn" onClick={exportPdf}>⬇ Case file (PDF)</button>
                    </div>
                    <div style={{ color: 'var(--muted)', fontSize: 11.5, marginTop: 6 }}>
                      rollup = min-hop-weighted: a single weak hop pulls the score down
                    </div>
                  </div>

                  <div className="card">
                    <h3>Detection snapshots</h3>
                    <div className="det-grid">
                      {trajectory.detections.map((d, i) => (
                        <div
                          className={`det-chip ${selectedDet === d.det_id ? 'selected' : ''}`}
                          key={d.det_id}
                          onClick={() => setSelectedDet(d.det_id)}
                        >
                          {d.crop_url
                            ? <img src={d.crop_url} alt={d.camera_id} />
                            : <div className="noimg">no crop stored</div>}
                          <div className="cam">{i + 1}. {d.camera_id}</div>
                          <div className="t">{new Date(d.ts).toLocaleTimeString()}</div>
                          <div className="t" style={{ fontFamily: 'ui-monospace, monospace' }}>
                            {d.plate_raw || '—'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="card">
                    <h3>Hop evidence (hover nodes on map too)</h3>
                    {trajectory.hops.map(h => {
                      const from = trajectory.detections.find(d => d.det_id === h.from_det)
                      const to = trajectory.detections.find(d => d.det_id === h.to_det)
                      return (
                        <div
                          key={h.seq}
                          className={`hop ${h.anomaly ? 'anomaly' : ''} ${selectedDet === h.to_det ? 'selected' : ''}`}
                          onClick={() => setSelectedDet(h.to_det)}
                        >
                          <div>
                            <span className={`tier tier-${h.tier}`}>{h.tier}</span>
                            <b>{from?.camera_id} → {to?.camera_id}</b>{' '}
                            <span className={`conf-badge ${confClass(h.confidence)}`}>
                              {Math.round(h.confidence * 100)}%
                            </span>
                          </div>
                          <div className="reason">{h.reason}</div>
                          <div className="signals">
                            {h.edit_distance != null && <span className="sig">ed={h.edit_distance}</span>}
                            {h.appearance_score != null && <span className="sig">cos={h.appearance_score.toFixed(2)}</span>}
                            {h.time_score != null && <span className="sig">t={h.time_score.toFixed(2)}</span>}
                            {h.ocr_conf_min != null && <span className="sig">ocr={h.ocr_conf_min.toFixed(2)}</span>}
                            {h.hop_reach === 2 && <span className="sig">2-hop (camera missed)</span>}
                            {h.anomaly && <span className="sig" style={{ color: 'var(--red)' }}>{h.anomaly}</span>}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </>
              ) : (
                <div className="card">
                  <h3>Trajectory</h3>
                  <div className="loading">
                    Search a plate (e.g. <b>KA01VX2026</b>) or pick a Demo scenario →
                  </div>
                </div>
              )}
            </>
          )}

          {tab === 'alerts' && (
            <div className="card">
              <h3>Alerts (latest {alerts.length})</h3>
              {alerts.map(a => (
                <div className={`alert ${a.kind === 'soft_anomaly' ? 'soft' : ''}`} key={a.alert_id}>
                  <b>{a.kind.replace('_', ' ')}</b> · {a.severity} · {a.plate}
                  <div>{a.title}</div>
                  <div className="ev">{JSON.stringify(a.evidence, null, 1)}</div>
                </div>
              ))}
              {!alerts.length && <div className="loading">no alerts — POST /api/alerts/evaluate to run detection</div>}
            </div>
          )}

          {tab === 'analytics' && (
            <div className="card">
              <h3>Analyst view — traffic analytics</h3>
              <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 10 }}>
                Toggle the density heatmap on the map. Segment speeds and route density:
              </div>
              <button className="btn" onClick={() => setShowHeat(!showHeat)}>
                {showHeat ? 'Hide density heatmap' : 'Show density heatmap'}
              </button>
              <div style={{ marginTop: 12, fontSize: 12 }}>
                <b>API endpoints for detailed analysis:</b>
                <div style={{ fontFamily: 'ui-monospace, monospace', color: 'var(--muted)', marginTop: 6, lineHeight: 1.8 }}>
                  GET /api/analytics/heatmap<br />
                  GET /api/analytics/segment-speeds<br />
                  GET /api/analytics/route-density<br />
                  GET /api/whatif/candidates — camera placement planning<br />
                </div>
              </div>
            </div>
          )}

          {tab === 'compare' && (
            <ComparePanel
              onCompare={runCompare}
              compareResult={compareResult}
              comparePlates={comparePlates}
              setComparePlates={setComparePlates}
            />
          )}

          {tab === 'whatif' && (
            <WhatIfPanel
              onEvaluate={evaluateWhatIf}
              candidates={whatifCandidates}
              result={whatifResult}
              onPickCandidate={(lat, lng) => evaluateWhatIf(lat, lng)}
              busy={whatifBusy}
            />
          )}

          {tab === 'demo' && (
            <div className="card">
              <h3>Demo scenarios</h3>
              {scenarios.map(s => (
                <button
                  key={s.key}
                  className={`scenario ${activeSession === s.key ? 'active' : ''}`}
                  onClick={() => runScenario(s)}
                >
                  <b>{s.title}</b>
                  {s.payload?.narrative && (
                    <div className="narrative">👁 {s.payload.narrative}</div>
                  )}
                  <div className="desc">{s.description}</div>
                </button>
              ))}
            </div>
          )}

          {tab === 'audit' && <AuditPanel />}
          {tab === 'tuning' && <TuningPanel />}
          {tab === 'notify' && <NotifyPanel />}
        </aside>
      </div>
    </div>
  )
}
