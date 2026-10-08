import { useEffect, useState } from 'react'

// ---------- Compare view ----------
export function ComparePanel({ onCompare, compareResult, comparePlates, setComparePlates }) {
  const [input, setInput] = useState('')

  const add = () => {
    const p = input.trim().toUpperCase()
    if (p && !comparePlates.includes(p) && comparePlates.length < 6) {
      setComparePlates([...comparePlates, p])
    }
    setInput('')
  }
  const remove = (p) => setComparePlates(comparePlates.filter(x => x !== p))
  const run = () => comparePlates.length >= 2 && onCompare()

  return (
    <div className="card">
      <h3>Multi-vehicle comparison</h3>
      <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
        <input
          style={{ flex: 1, background: 'var(--bg)', border: '1px solid var(--border)',
                   color: 'var(--text)', borderRadius: 6, padding: '6px 10px',
                   fontFamily: 'ui-monospace, monospace', fontSize: 12 }}
          placeholder="add plate (e.g. KA09ST0555)"
          value={input}
          onChange={e => setInput(e.target.value.toUpperCase())}
          onKeyDown={e => e.key === 'Enter' && add()}
        />
        <button className="btn" onClick={add}>Add</button>
        <button className="btn" disabled={comparePlates.length < 2}
                style={{ opacity: comparePlates.length < 2 ? 0.5 : 1 }}
                onClick={run}>Compare</button>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
        {comparePlates.map((p, i) => (
          <span key={p} className="cmp-chip" style={{ borderColor: CMP_COLORS[i % 6] }}>
            <span style={{ color: CMP_COLORS[i % 6] }}>●</span> {p}
            <span className="rm" onClick={() => remove(p)}>✕</span>
          </span>
        ))}
        {!comparePlates.length && <span style={{ color: 'var(--muted)', fontSize: 12 }}>add 2–6 plates</span>}
      </div>
      {compareResult && (
        <>
          <h3>Shared timeline</h3>
          <Timeline compareResult={compareResult} />
          {compareResult.convoys?.length > 0 && (
            <>
              <h3 style={{ marginTop: 10 }}>Co-location / convoy evidence</h3>
              {compareResult.convoys.map((c, i) => (
                <div key={i} className={`alert ${c.is_convoy ? '' : 'soft'}`} style={{ fontSize: 12 }}>
                  <b>{c.plates.join(' + ')}</b> {c.is_convoy ? '— CONVOY PATTERN' : '— co-located'}
                  <div style={{ margin: '4px 0' }}>{c.explanation}</div>
                  {c.meetings.slice(0, 5).map((m, j) => (
                    <div key={j} style={{ color: 'var(--muted)', fontSize: 11, fontFamily: 'ui-monospace, monospace' }}>
                      {m.camera_id} · gap {Math.round(m.gap_s)}s · {m.ts.slice(11, 19)}
                    </div>
                  ))}
                </div>
              ))}
            </>
          )}
        </>
      )}
    </div>
  )
}

function Timeline({ compareResult }) {
  const [t0, t1] = (() => {
    const all = compareResult.trajectories.flatMap(t => t.detections.map(d => new Date(d.ts).getTime()))
    return all.length ? [Math.min(...all), Math.max(...all)] : [0, 1]
  })()
  const span = Math.max(t1 - t0, 1)
  return (
    <div className="timeline">
      {compareResult.trajectories.map((t, i) => (
        <div key={t.plate} className="tl-row">
          <span className="tl-label" style={{ color: CMP_COLORS[i % 6] }}>{t.plate.slice(-4)}</span>
          <div className="tl-track">
            {t.detections.map((d, j) => (
              <span key={j} className="tl-dot" title={`${d.camera_id} ${d.ts.slice(11, 19)}`}
                    style={{
                      left: `${(new Date(d.ts).getTime() - t0) / span * 100}%`,
                      background: CMP_COLORS[i % 6],
                    }} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

export const CMP_COLORS = ['#58a6ff', '#3fb950', '#bc8cff', '#f85149', '#d29922', '#39c5cf']

// ---------- What-if placement ----------
export function WhatIfPanel({ onEvaluate, candidates, result, onPickCandidate, busy }) {
  const activeLat = result && !result.error ? result.hypothetical_camera?.lat : null
  const activeLng = result && !result.error ? result.hypothetical_camera?.lng : null
  const isActive = (c) =>
    activeLat != null && Math.abs(c.lat - activeLat) < 1e-4 && Math.abs(c.lng - activeLng) < 1e-4
  return (
    <div className="card">
      <h3>What-if: camera placement (city planning)</h3>
      <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 8 }}>
        Click anywhere on the map (or pick a candidate junction below). VANTRA re-stitches
        the hardest benchmark trips — those left with unlinkable gaps by two adjacent
        camera misses — with a hypothetical camera at that spot, and shows what it would close.
      </div>
      <div style={{ marginBottom: 10 }}>
        {candidates.map(c => (
          <button key={c.id} className="btn"
                  style={{ marginBottom: 6, fontSize: 11, width: '100%',
                           borderColor: isActive(c) ? 'var(--accent)' : 'var(--border)',
                           color: isActive(c) ? 'var(--accent)' : 'var(--text)',
                           display: 'block', textAlign: 'left' }}
                  onClick={() => onPickCandidate(c.lat, c.lng)}>
            {c.name}
            <div style={{ color: 'var(--muted)', fontSize: 10, marginTop: 2 }}>
              {c.note || (c.edge_distance_m ? Math.round(c.edge_distance_m) + 'm edge' : '')}
            </div>
          </button>
        ))}
      </div>
      {busy && <div className="loading">simulating…</div>}
      {result && result.error && <div className="alert">{result.error}</div>}
      {result && !result.error && result.hypothetical_camera && result.before && result.after && (
        <div>
          <div style={{ fontSize: 12, marginBottom: 8 }}>
            Hypothetical camera at {result.hypothetical_camera.lat.toFixed(4)},
            {' '}{result.hypothetical_camera.lng.toFixed(4)} — would connect to{' '}
            <b>{result.hypothetical_camera.connects_to.join(', ')}</b>
          </div>
          <table className="before-after">
            <thead><tr><th></th><th>without</th><th>with new camera</th></tr></thead>
            <tbody>
              <tr><td>full-path stitching (all trips)</td>
                  <td>{result.before.full_pct}%</td>
                  <td><b>{result.after.full_pct}%</b></td></tr>
              <tr><td>hops linked</td>
                  <td>{result.before.link_pct}%</td>
                  <td><b>{result.after.link_pct}%</b></td></tr>
              {result.affected_trips?.n_affected_trips > 0 && (
                <tr><td>trips through this corridor (previously broken)</td>
                    <td>{result.affected_trips.full_pct_before}%</td>
                    <td><b>{result.affected_trips.full_pct_after}%</b></td></tr>
              )}
            </tbody>
          </table>
          {result.affected_trips?.n_affected_trips > 0 ? (
            <div style={{ color: 'var(--green)', fontSize: 12, marginTop: 6 }}>
              ✓ would close {result.affected_trips.n_affected_trips} currently-unlinkable trip
              {result.affected_trips.n_affected_trips > 1 ? 's' : ''} passing this corridor
            </div>
          ) : (
            <div style={{ color: 'var(--muted)', fontSize: 12, marginTop: 6 }}>
              no currently-broken trips would cross this location in the benchmark sample
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------- Tuning panel ----------
export function TuningPanel() {
  const [cfg, setCfg] = useState(null)
  const [preview, setPreview] = useState(null)
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)
  const [adminKey, setAdminKey] = useState('vantra-admin')
  const [saveError, setSaveError] = useState(null)

  useEffect(() => {
    fetch('/api/tuning').then(r => r.json()).then(d => setCfg(d.effective))
  }, [])

  const update = (k, v) => {
    setCfg({ ...cfg, [k]: v })
    setSaved(false)
    // debounce preview
    clearTimeout(update._t)
    update._t = setTimeout(() => runPreview({ ...cfg, [k]: v }), 600)
  }
  const runPreview = async (values) => {
    setBusy(true)
    const d = await fetch('/api/tuning/preview', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(values),
    }).then(r => r.json())
    setPreview(d); setBusy(false)
  }
  const save = async () => {
    const values = {}
    for (const k of ['fuzzy_ed_high_conf', 'fuzzy_ed_mid_conf', 'fuzzy_ed_low_conf',
                     'appearance_floor', 'appearance_margin'])
      values[k] = cfg[k]
    const r = await fetch('/api/tuning/save', {
      method: 'POST', headers: {
        'content-type': 'application/json',
        'x-vantra-admin-key': adminKey || 'vantra-admin',
      },
      body: JSON.stringify({ values }),
    })
    if (r.ok) setSaved(true)
    else setSaveError((await r.json()).detail || 'save failed')
  }
  const reset = async () => {
    await fetch('/api/tuning/reset', {
      method: 'POST', headers: { 'x-vantra-admin-key': adminKey || 'vantra-admin' },
    })
    const d = await fetch('/api/tuning').then(r => r.json())
    setCfg(d.effective); setPreview(null); setSaved(false)
  }

  if (!cfg) return <div className="loading">loading…</div>
  const sliders = [
    ['fuzzy_ed_high_conf', 'max edit distance @ high OCR conf (≥0.88)', 0, 3, 1,
     'how many wrong characters a confident read may still match'],
    ['fuzzy_ed_mid_conf', 'max edit distance @ mid OCR conf (≥0.62)', 0, 4, 1,
     'threshold for average-confidence reads'],
    ['fuzzy_ed_low_conf', 'max edit distance @ low OCR conf', 0, 5, 1,
     'wider window for badly-read plates — more leads, more risk'],
    ['appearance_floor', 'appearance (Re-ID cosine) floor', 0.3, 0.95, 0.01,
     'minimum vehicle-similarity to link a hop by appearance alone'],
    ['appearance_margin', 'ambiguity margin (refuse if top-2 closer than this)', 0.01, 0.3, 0.01,
     'bigger = more matches refused for human review'],
  ]
  return (
    <div className="card">
      <h3>Matching thresholds — operator tuning</h3>
      <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 10 }}>
        <b style={{ color: 'var(--green)' }}>Strict</b> (tight thresholds) = evidentiary use,
        fewer but defensible matches. <b style={{ color: 'var(--yellow)' }}>Loose</b> =
        investigative leads, more matches, more false links. Previews are dry-runs —
        nothing changes until you save.
      </div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
        <button className="btn" onClick={() => {
          const strict = { fuzzy_ed_high_conf: 0, fuzzy_ed_mid_conf: 1, fuzzy_ed_low_conf: 1,
                           appearance_floor: 0.8, appearance_margin: 0.15 }
          setCfg({ ...cfg, ...strict }); runPreview(strict)
        }}>Preset: strict (evidentiary)</button>
        <button className="btn" onClick={() => {
          const loose = { fuzzy_ed_high_conf: 2, fuzzy_ed_mid_conf: 3, fuzzy_ed_low_conf: 4,
                          appearance_floor: 0.4, appearance_margin: 0.03 }
          setCfg({ ...cfg, ...loose }); runPreview(loose)
        }}>Preset: loose (investigative)</button>
      </div>
      {sliders.map(([k, label, min, max, step, hint]) => (
        <div key={k} style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 12, display: 'flex', justifyContent: 'space-between' }}>
            <span>{label}</span><b>{cfg[k]}</b>
          </div>
          <input type="range" min={min} max={max} step={step} value={cfg[k]}
                 style={{ width: '100%' }}
                 onChange={e => update(k, parseFloat(e.target.value))} />
          <div style={{ color: 'var(--muted)', fontSize: 11 }}>{hint}</div>
        </div>
      ))}
      {busy && <div className="loading">running dry-run on history…</div>}
      {preview && !busy && (
        <table className="before-after">
          <thead><tr><th></th><th>current</th><th>with your values</th><th>Δ</th></tr></thead>
          <tbody>
            {['exact', 'fuzzy_unique', 'appearance_only'].map(t => (
              <tr key={t}>
                <td>{t.replace('_', ' ')} links</td>
                <td>{preview.before.tiers[t] || 0}</td>
                <td><b>{preview.after.tiers[t] || 0}</b></td>
                <td style={{ color: (preview.after.tiers[t] || 0) - (preview.before.tiers[t] || 0) > 0 ? 'var(--green)' : 'var(--red)' }}>
                  {(preview.after.tiers[t] || 0) - (preview.before.tiers[t] || 0) > 0 ? '+' : ''}
                  {(preview.after.tiers[t] || 0) - (preview.before.tiers[t] || 0)}
                </td>
              </tr>
            ))}
            <tr><td>hard anomalies</td>
                <td>{preview.before.hard_anomalies}</td>
                <td><b>{preview.after.hard_anomalies}</b></td>
                <td>{preview.delta_anomalies > 0 ? '+' : ''}{preview.delta_anomalies}</td></tr>
            <tr><td>mean trajectory confidence</td>
                <td>{preview.before.mean_conf}</td>
                <td><b>{preview.after.mean_conf}</b></td>
                <td></td></tr>
          </tbody>
        </table>
      )}
      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>
          admin API key (saving requires it — demo default: vantra-admin)
        </div>
        <input value={adminKey} onChange={e => setAdminKey(e.target.value)}
               style={{ width: 200, background: 'var(--bg)', color: 'var(--text)',
                        border: '1px solid var(--border)', borderRadius: 6,
                        padding: 6, fontSize: 12, fontFamily: 'ui-monospace, monospace' }} />
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        <button className="btn" onClick={save}>Save as production</button>
        <button className="btn" onClick={reset}>Reset to defaults</button>
        {saved && <span style={{ color: 'var(--green)', fontSize: 12, alignSelf: 'center' }}>saved ✓</span>}
        {saveError && <span style={{ color: 'var(--red)', fontSize: 12, alignSelf: 'center' }}>{saveError}</span>}
      </div>
    </div>
  )
}

// ---------- Audit log ----------
export function AuditPanel() {
  const [entries, setEntries] = useState([])
  const [ops, setOps] = useState([])
  const [op, setOp] = useState('')
  const [date, setDate] = useState('')

  const load = async () => {
    const q = new URLSearchParams()
    if (op) q.set('op', op)
    if (date) q.set('date', date)
    const d = await fetch(`/api/audit?${q}`).then(r => r.json())
    setEntries(d.entries || []); setOps(d.operators || [])
  }
  useEffect(() => { load() }, [])

  return (
    <div className="card">
      <h3>Audit log — who queried what, when</h3>
      <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
        <select value={op} onChange={e => setOp(e.target.value)}
                style={{ background: 'var(--bg)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 6, padding: 6 }}>
          <option value="">all operators</option>
          {ops.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <input type="date" value={date} onChange={e => setDate(e.target.value)}
               style={{ background: 'var(--bg)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 6, padding: 6 }} />
        <button className="btn" onClick={load}>Filter</button>
      </div>
      <div style={{ maxHeight: 420, overflowY: 'auto' }}>
        <table className="audit-table">
          <thead><tr><th>when</th><th>operator</th><th>action</th><th>detail</th></tr></thead>
          <tbody>
            {entries.map(e => (
              <tr key={e.id}>
                <td>{new Date(e.ts).toLocaleString()}</td>
                <td>{e.operator}</td>
                <td>{e.action}</td>
                <td style={{ fontFamily: 'ui-monospace, monospace' }}>{e.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!entries.length && <div className="loading">no entries</div>}
      </div>
    </div>
  )
}

// ---------- Notify config ----------
export function NotifyPanel() {
  const [cfg, setCfg] = useState(null)
  const [testResult, setTestResult] = useState(null)

  useEffect(() => {
    fetch('/api/notify/config').then(r => r.json()).then(setCfg)
  }, [])

  const save = async () => {
    const d = await fetch('/api/notify/config', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ webhook_url: cfg.webhook_url || null,
                             email: cfg.email || null, enabled: cfg.enabled }),
    }).then(r => r.json())
    setCfg(d)
  }
  const test = async () => {
    const d = await fetch('/api/notify/test', { method: 'POST' }).then(r => r.json())
    setTestResult(d)
  }

  if (!cfg) return <div className="loading">loading…</div>
  return (
    <div className="card">
      <h3>Notification hooks</h3>
      <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 10 }}>
        Blacklist hits and hard anomalies fire this webhook (stubbed — logs
        "would notify" if no URL set). Integration point for SMS/email/dispatch systems.
      </div>
      <label style={{ fontSize: 12, display: 'block', marginBottom: 6 }}>
        Webhook URL
        <input style={{ width: '100%', marginTop: 4, background: 'var(--bg)',
                        border: '1px solid var(--border)', color: 'var(--text)',
                        borderRadius: 6, padding: 6, fontFamily: 'ui-monospace, monospace' }}
               placeholder="https://hooks.example.com/vantra"
               value={cfg.webhook_url || ''}
               onChange={e => setCfg({ ...cfg, webhook_url: e.target.value })} />
      </label>
      <label style={{ fontSize: 12, display: 'flex', gap: 6, alignItems: 'center', marginBottom: 10 }}>
        <input type="checkbox" checked={cfg.enabled}
               onChange={e => setCfg({ ...cfg, enabled: e.target.checked })} />
        enabled
      </label>
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="btn" onClick={save}>Save</button>
        <button className="btn" onClick={test}>Send test</button>
      </div>
      {testResult && (
        <div style={{ marginTop: 8, fontSize: 12, color: 'var(--muted)' }}>
          test fired — check the API server log for the notify line
          (endpoint: {testResult.config?.webhook_url || 'none configured'})
        </div>
      )}
    </div>
  )
}
