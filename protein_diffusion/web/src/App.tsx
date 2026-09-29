import { useState } from 'react'
import './App.css'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

interface Variant {
  variant: string
  fitness: number | null
  status: 'measured' | 'unmeasured'
}

interface SampleResponse {
  n: number
  measured: number
  variants: Variant[]
}

function App() {
  const [n, setN] = useState(32)
  const [seed, setSeed] = useState(7)
  const [cond, setCond] = useState('2.0')
  const [guidance, setGuidance] = useState(4.0)
  const [result, setResult] = useState<SampleResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function sample() {
    setLoading(true)
    setError(null)
    try {
      const body: Record<string, number> = { n, seed, guidance }
      if (cond.trim() !== '') body.cond = Number(cond)
      const r = await fetch(`${API}/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!r.ok) throw new Error(`API ${r.status}: ${await r.text()}`)
      setResult(await r.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const measuredVars = result?.variants.filter((v) => v.status === 'measured') ?? []
  const meanAll =
    result && result.n > 0
      ? result.variants.reduce((s, v) => s + (v.fitness ?? 0), 0) / result.n
      : null

  return (
    <main>
      <h1>protein-diffusion variant browser</h1>
      <p className="sub">
        Conditional DDPM over the GB1 landscape. Sampling runs on the FastAPI
        service (<code>{API}</code>); fitness values come from the measured
        oracle — variants outside the landscape are flagged{' '}
        <em>unmeasured</em> and count as zero in the aggregate.
      </p>

      <section className="controls">
        <label>
          n
          <input type="number" min={1} max={512} value={n}
            onChange={(e) => setN(Number(e.target.value))} />
        </label>
        <label>
          seed
          <input type="number" min={0} value={seed}
            onChange={(e) => setSeed(Number(e.target.value))} />
        </label>
        <label>
          cond (log1p fitness, blank = unconditional)
          <input value={cond}
            onChange={(e) => setCond(e.target.value)} />
        </label>
        <label>
          guidance w: {guidance.toFixed(1)}
          <input type="range" min={0} max={16} step={0.5} value={guidance}
            onChange={(e) => setGuidance(Number(e.target.value))} />
        </label>
        <button onClick={sample} disabled={loading}>
          {loading ? 'sampling…' : 'sample'}
        </button>
      </section>

      {error && <p className="error">{error}</p>}

      {result && (
        <>
          <section className="stats">
            <div><strong>{result.n}</strong><span>variants</span></div>
            <div><strong>{result.measured}</strong><span>measured ({((result.measured / result.n) * 100).toFixed(0)}%)</span></div>
            <div><strong>{meanAll?.toFixed(3)}</strong><span>mean fitness (unmeasured = 0)</span></div>
            <div>
              <strong>{measuredVars.length > 0
                ? Math.max(...measuredVars.map((v) => v.fitness ?? 0)).toFixed(3)
                : '—'}</strong>
              <span>best measured</span>
            </div>
          </section>
          <table>
            <thead>
              <tr><th>#</th><th>variant</th><th>fitness</th><th>status</th></tr>
            </thead>
            <tbody>
              {result.variants.map((v, i) => (
                <tr key={i} className={v.status}>
                  <td>{i + 1}</td>
                  <td className="mono">{v.variant}</td>
                  <td>{v.fitness !== null ? v.fitness.toFixed(4) : '—'}</td>
                  <td>{v.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </main>
  )
}

export default App
