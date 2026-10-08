import { useEffect, useRef, useState } from 'react'

const SPEEDS = [1, 4, 8, 30, 120]

export default function LivePanel({ onEvents, onAlerts }) {
  const [mode, setMode] = useState('replay')   // 'replay' | 'video'
  const [session, setSession] = useState(null)
  const [running, setRunning] = useState(false)
  const [speed, setSpeed] = useState(8)
  const [progress, setProgress] = useState(0)
  const [simTime, setSimTime] = useState(null)
  const [nDets, setNDets] = useState(0)
  const [alerts, setAlerts] = useState([])
  const [feed, setFeed] = useState([])
  const [videoStats, setVideoStats] = useState(null)
  const timer = useRef(null)
  const knownIds = useRef(new Set())

  useEffect(() => () => clearInterval(timer.current), [])

  const start = async (spd = 8) => {
    clearInterval(timer.current)
    const r = await fetch('/api/live/session', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ speed: spd, minutes: 60 }),
    }).then(r => r.json())
    if (!r.session_id) return
    setSession(r.session_id)
    setRunning(true)
    setProgress(0); setNDets(0); setAlerts([]); setFeed([])
    knownIds.current = new Set()
    timer.current = setInterval(() => poll(r.session_id), 1000)
  }

  const startVideo = async () => {
    clearInterval(timer.current)
    const r = await fetch('/api/video-live/session', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ video: 'cam_traffic.mp4', camera_id: 'CAM03', sample_every: 15 }),
    }).then(r => r.json())
    if (!r.session_id) return
    setSession(r.session_id)
    setRunning(true); setProgress(0); setNDets(0); setAlerts([]); setFeed([]); setVideoStats(null)
    timer.current = setInterval(() => pollVideo(r.session_id), 1500)
  }

  const pollVideo = async (sid) => {
    try {
      const d = await fetch(`/api/video-live/session/${sid}/tick`).then(r => r.json())
      if (d.detail) return
      setRunning(d.running)
      setProgress(d.total_frames ? d.frame / d.total_frames : 0)
      setNDets(d.n_detections ?? 0)
      setVideoStats({
        frame: d.frame, total: d.total_frames,
        ms: d.pipeline_ms_per_frame, eff: d.effective_rate_fps, native: d.native_fps,
      })
      if (d.new_events?.length) {
        const marked = d.new_events.map(ev => ({
          ...ev,
          lat: ev.lat ?? 12.961, lng: ev.lng ?? 77.6385,   // CAM03 position for map pin
          is_new: true,
        }))
        onEvents?.(marked, d)
        setFeed(prev => [...prev, ...marked].slice(-25))
        setTimeout(() => setFeed(prev => prev.map(f => ({ ...f, is_new: false }))), 2000)
      }
      if (d.done || d.running === false) clearInterval(timer.current)
    } catch { /* ignore */ }
  }
  const poll = async (sid) => {
    try {
      const d = await fetch(`/api/live/session/${sid}/tick`).then(r => r.json())
      if (d.error || d.detail) return
      setRunning(d.running)
      setProgress(d.progress ?? 0)
      setSimTime(d.sim_time)
      setNDets(d.n_detections ?? 0)
      if (d.alerts?.length) {
        setAlerts(d.alerts)
        onAlerts?.(d.alerts)
      }
      if (d.new_events?.length) {
        const marked = d.new_events.map(ev => ({ ...ev, is_new: true }))
        onEvents?.(marked, d)
        setFeed(prev => [...prev, ...marked].slice(-25))
        // fade the "new" highlight after a moment
        setTimeout(() => setFeed(prev => prev.map(f => ({ ...f, is_new: false }))), 2000)
      }
      if (d.done || d.running === false) clearInterval(timer.current)
    } catch { /* server restarting etc */ }
  }

  const control = async (action, spd) => {
    if (!session) return
    if (action === 'pause') clearInterval(timer.current)
    if (action === 'resume' && running) timer.current = setInterval(() => poll(session), 1000)
    const d = await fetch(`/api/live/session/${session}/control?action=${action}${spd ? `&speed=${spd}` : ''}`, {
      method: 'POST',
    }).then(r => r.json())
    if (action === 'pause') setRunning(false)
    if (action === 'resume') setRunning(true)
    if (action === 'speed' && spd) {
      setSpeed(spd)
      if (running) { clearInterval(timer.current); timer.current = setInterval(() => poll(session), 1000) }
    }
  }

  const restart = () => {
    clearInterval(timer.current)
    setSession(null); setRunning(false); setFeed([]); setVideoStats(null); setProgress(0)
  }

  return (
    <div className="card">
      <h3>Live mode</h3>
      {!session && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
          <button className="btn" style={{
            flex: 1, borderColor: mode === 'replay' ? 'var(--accent)' : 'var(--border)',
            color: mode === 'replay' ? 'var(--accent)' : 'var(--text)',
          }} onClick={() => setMode('replay')}>JSON replay (default)</button>
          <button className="btn" style={{
            flex: 1, borderColor: mode === 'video' ? 'var(--accent)' : 'var(--border)',
            color: mode === 'video' ? 'var(--accent)' : 'var(--text)',
          }} onClick={() => setMode('video')}>Video pipeline</button>
        </div>
      )}
      {!session && mode === 'replay' && (
        <>
          <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 8 }}>
            Replays the busiest hour of simulated city traffic at real-world pace.
            Detections appear on the map as cameras see them; trajectories link live.
          </div>
          <button className="btn" style={{ width: '100%' }} onClick={() => start(8)}>
            ▶ Start live replay (8×)
          </button>
        </>
      )}
      {!session && mode === 'video' && (
        <>
          <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 8 }}>
            Decodes an actual traffic-camera video clip frame by frame and runs the
            REAL detection+OCR pipeline on each sampled frame (~0.6fps effective —
            honest rate, not real-time).
          </div>
          <button className="btn" style={{ width: '100%' }} onClick={startVideo}>
            ▶ Start video detection (CAM03)
          </button>
        </>
      )}
      {session && (
        <>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 8 }}>
            <button className="btn" onClick={() => running ? control('pause') : control('resume')}>
              {running ? '⏸ Pause' : '▶ Resume'}
            </button>
            {videoStats ? (
              <span style={{ fontSize: 12, color: 'var(--muted)' }}>
                frame {videoStats.frame}/{videoStats.total} · {nDets} reads · {videoStats.eff}fps eff (native {videoStats.native})
              </span>
            ) : (
              <span style={{ fontSize: 12, color: 'var(--muted)' }}>
                {simTime ? new Date(simTime).toLocaleTimeString() : ''} · {nDets} dets
              </span>
            )}
            <span style={{ flex: 1 }} />
            <button className="btn" onClick={restart}>✕ New session</button>
          </div>
          {!videoStats && (
            <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
              {SPEEDS.map(s => (
                <button key={s} className="btn" style={{
                  flex: 1, padding: '4px 0', fontSize: 11,
                  borderColor: speed === s ? 'var(--accent)' : 'var(--border)',
                  color: speed === s ? 'var(--accent)' : 'var(--text)',
                }} onClick={() => control('speed', s)}>{s}×</button>
              ))}
            </div>
          )}
          <div style={{ height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden', marginBottom: 10 }}>
            <div style={{ width: `${progress * 100}%`, height: '100%', background: 'var(--accent)', transition: 'width .5s' }} />
          </div>
          {alerts.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              {alerts.slice(-4).map((a, i) => (
                <div key={i} className="alert" style={{ padding: '5px 8px', fontSize: 12 }}>
                  <b>{a.kind.replace('_', ' ')}</b> · {a.title}
                </div>
              ))}
            </div>
          )}
          <div style={{ maxHeight: 220, overflowY: 'auto' }}>
            {feed.length === 0 && <div className="loading" style={{ padding: 8 }}>waiting for detections…</div>}
            {feed.map((f, i) => (
              <div key={i} className="live-feed-row" style={{ opacity: f.is_new ? 1 : 0.55 }}>
                <span className="cam">{f.camera_id}</span>
                <span className="plate">{f.plate || '—'}</span>
                <span className="t">{f.ts.slice(11, 19)}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
