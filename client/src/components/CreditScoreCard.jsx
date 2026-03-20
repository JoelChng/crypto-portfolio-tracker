/* CreditScoreCard — displays ML 0-1000 credit score, grade A-E, SHAP reason codes */

const GRADE_META = {
  A: { color: '#22c55e', bg: 'rgba(34,197,94,0.1)',   label: 'Excellent',  desc: 'Very low default risk (PD < 2%)' },
  B: { color: '#14b8a6', bg: 'rgba(20,184,166,0.1)',  label: 'Good',       desc: 'Low default risk (PD 2–5%)' },
  C: { color: '#eab308', bg: 'rgba(234,179,8,0.1)',   label: 'Fair',       desc: 'Moderate risk (PD 5–12%)' },
  D: { color: '#f97316', bg: 'rgba(249,115,22,0.1)',  label: 'Poor',       desc: 'High risk (PD 12–25%)' },
  E: { color: '#ef4444', bg: 'rgba(239,68,68,0.1)',   label: 'Very High',  desc: 'Very high risk (PD > 25%)' },
}

function ScoreArc({ score }) {
  const SIZE    = 160
  const SW      = 14
  const r       = (SIZE - SW) / 2
  const circum  = 2 * Math.PI * r
  const cx      = SIZE / 2
  const cy      = SIZE / 2
  const pct     = Math.min(score / 1000, 1)

  // Colour transitions: red → orange → yellow → teal → green
  function arcColor(p) {
    if (p < 0.3)  return '#ef4444'
    if (p < 0.55) return '#f97316'
    if (p < 0.73) return '#eab308'
    if (p < 0.82) return '#14b8a6'
    return '#22c55e'
  }

  const color = arcColor(pct)
  const dash  = pct * circum

  return (
    <div className="relative flex items-center justify-center" style={{ width: SIZE, height: SIZE }}>
      <svg width={SIZE} height={SIZE} style={{ transform: 'rotate(-90deg)', position: 'absolute', top: 0, left: 0 }}>
        {/* Track */}
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border-card)" strokeWidth={SW} />
        {/* Fill */}
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={color}
          strokeWidth={SW}
          strokeDasharray={`${dash} ${circum - dash}`}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 6px ${color}80)` }}
        />
      </svg>
      <div className="text-center relative z-10">
        <div className="text-3xl font-black text-white leading-none">{score}</div>
        <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>/ 1000</div>
      </div>
    </div>
  )
}

function ReasonCode({ code, text, direction }) {
  const isPos = direction === 'risk_decrease'
  return (
    <div
      className={`px-3 py-2 rounded-lg text-xs leading-snug ${isPos ? 'reason-positive' : 'reason-negative'}`}
      style={{ backgroundColor: isPos ? 'rgba(34,197,94,0.05)' : 'rgba(239,68,68,0.05)' }}
    >
      <span className="font-semibold mr-1.5" style={{ color: isPos ? '#22c55e' : '#ef4444' }}>
        {isPos ? '▲' : '▼'} {code}
      </span>
      <span style={{ color: 'var(--text-secondary)' }}>{text}</span>
    </div>
  )
}

/* ── Offline / loading states ── */
function OfflineState() {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <div className="text-3xl mb-3">🤖</div>
      <p className="text-sm font-medium text-white mb-1">ML Scoring API Offline</p>
      <p className="text-xs leading-relaxed max-w-48" style={{ color: 'var(--text-muted)' }}>
        Start the FastAPI server on :8000 to enable ML credit scoring.
      </p>
      <code className="mt-3 px-3 py-1.5 rounded text-xs" style={{ backgroundColor: 'rgba(255,255,255,0.05)', color: '#14b8a6' }}>
        make api
      </code>
    </div>
  )
}

function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center py-8 gap-3">
      <div className="w-10 h-10 rounded-full border-2 border-purple-500 border-t-transparent animate-spin" />
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Computing ML credit score…</p>
    </div>
  )
}

export default function CreditScoreCard({ creditScore, loading }) {
  const grade = creditScore?.risk_grade
  const meta  = grade ? GRADE_META[grade] : null

  // API returns "risk_decrease" (good) and "risk_increase" (bad)
  const positives = (creditScore?.top_reason_codes ?? []).filter(r => r.direction === 'risk_decrease').slice(0, 3)
  const negatives = (creditScore?.top_reason_codes ?? []).filter(r => r.direction === 'risk_increase').slice(0, 3)

  return (
    <div className="rounded-xl p-5 h-full flex flex-col card-glow"
         style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-semibold text-white text-sm">ML Credit Score</h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
            Random Forest ensemble · 76 features
          </p>
        </div>
        <span className="badge" style={{ backgroundColor: 'rgba(139,92,246,0.1)', color: '#8b5cf6', border: '1px solid rgba(139,92,246,0.2)' }}>
          🤖 ML
        </span>
      </div>

      {loading ? (
        <LoadingState />
      ) : !creditScore ? (
        <OfflineState />
      ) : (
        <div className="flex flex-col gap-4 flex-1">

          {/* Score arc + grade */}
          <div className="flex items-center gap-4">
            <ScoreArc score={creditScore.score ?? 0} />
            <div className="flex-1">
              {/* Grade badge */}
              {meta && (
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg mb-3"
                     style={{ backgroundColor: meta.bg, border: `1px solid ${meta.color}30` }}>
                  <span className="text-2xl font-black" style={{ color: meta.color }}>{grade}</span>
                  <div>
                    <div className="text-xs font-semibold text-white">{meta.label}</div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{meta.desc}</div>
                  </div>
                </div>
              )}

              {/* PD */}
              {creditScore.pd_90d != null && (
                <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  90-day default probability:{' '}
                  <span className="font-semibold text-white">
                    {(creditScore.pd_90d * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Reason codes */}
          {(positives.length > 0 || negatives.length > 0) && (
            <div>
              <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
                KEY FACTORS
              </p>
              <div className="space-y-1.5">
                {positives.map((rc, i) => (
                  <ReasonCode key={i} code={rc.code} text={rc.text} direction="positive" />
                ))}
                {negatives.map((rc, i) => (
                  <ReasonCode key={i} code={rc.code} text={rc.text} direction="negative" />
                ))}
              </div>
            </div>
          )}

          {/* Source note */}
          <div className="mt-auto pt-3 border-t" style={{ borderColor: 'var(--border-subtle)' }}>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Powered by{' '}
              <a href="https://github.com/JoelChng/onchain-credit-scoring"
                 target="_blank" rel="noreferrer"
                 className="hover:text-white transition-colors" style={{ color: '#8b5cf6' }}>
                onchain-credit-scoring
              </a>
              {' '}· SHAP-explained
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
