import { useEffect, useRef } from 'react'

const CATEGORIES = [
  { label: 'Conservative', colour: '#22c55e', emoji: '🟢', max: 30 },
  { label: 'Moderate',     colour: '#eab308', emoji: '🟡', max: 55 },
  { label: 'Aggressive',   colour: '#f97316', emoji: '🟠', max: 75 },
  { label: 'Very High Risk', colour: '#ef4444', emoji: '🔴', max: 100 },
]

function getCategory(score) {
  return CATEGORIES.find(c => score <= c.max) ?? CATEGORIES[3]
}

export default function RiskGauge({ risk }) {
  const canvasRef = useRef(null)
  const score     = risk ? Math.round(risk.compositeScore) : null
  const category  = score != null ? getCategory(score) : null

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || score == null) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    const cx = W / 2, cy = H * 0.72
    const R  = W * 0.36

    ctx.clearRect(0, 0, W, H)

    // Track
    ctx.beginPath()
    ctx.arc(cx, cy, R, Math.PI, 0)
    ctx.lineWidth  = 18
    ctx.strokeStyle = '#1e2a3a'
    ctx.lineCap    = 'round'
    ctx.stroke()

    // Coloured segments
    const segs = [
      { from: 0,  to: 30,  colour: '#22c55e' },
      { from: 30, to: 55,  colour: '#eab308' },
      { from: 55, to: 75,  colour: '#f97316' },
      { from: 75, to: 100, colour: '#ef4444' },
    ]
    segs.forEach(seg => {
      const aStart = Math.PI + (seg.from / 100) * Math.PI
      const aEnd   = Math.PI + (seg.to   / 100) * Math.PI
      ctx.beginPath()
      ctx.arc(cx, cy, R, aStart, aEnd)
      ctx.lineWidth  = 18
      ctx.strokeStyle = seg.colour
      ctx.lineCap    = 'butt'
      ctx.stroke()
    })

    // Active glow on the needle endpoint
    const angle = Math.PI + (score / 100) * Math.PI
    const nx = cx + (R - 10) * Math.cos(angle)
    const ny = cy + (R - 10) * Math.sin(angle)

    ctx.save()
    ctx.shadowColor = category?.colour ?? '#fff'
    ctx.shadowBlur  = 10

    // Needle
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.lineTo(nx, ny)
    ctx.lineWidth  = 3
    ctx.strokeStyle = '#ffffff'
    ctx.lineCap    = 'round'
    ctx.stroke()

    ctx.restore()

    // Centre dot
    ctx.beginPath()
    ctx.arc(cx, cy, 5, 0, 2 * Math.PI)
    ctx.fillStyle = '#ffffff'
    ctx.fill()
  }, [score, category])

  return (
    <div className="rounded-xl p-5 h-full flex flex-col card-glow"
         style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-semibold text-white text-sm">Risk Profile</h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
            Rule-based · 3 signals
          </p>
        </div>
        <span className="badge" style={{ backgroundColor: 'rgba(59,130,246,0.1)', color: '#3b82f6', border: '1px solid rgba(59,130,246,0.2)' }}>
          ⚡ Live
        </span>
      </div>

      {score == null ? (
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Risk data unavailable.</p>
      ) : (
        <>
          <canvas ref={canvasRef} width={260} height={155} className="w-full max-w-[260px] mx-auto block" />

          <div className="text-center mt-1 mb-4">
            <p className="text-4xl font-black text-white">{score}</p>
            <p className="text-sm font-semibold mt-0.5" style={{ color: category.colour }}>
              {category.emoji} {category.label}
            </p>
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>out of 100</p>
          </div>

          {/* Signal breakdown */}
          {risk.breakdown && (
            <div className="space-y-3 mt-auto">
              <SignalBar label="Holdings Composition"  score={risk.breakdown.holdingsScore}  weight="40%" explanation={risk.breakdown.holdingsExplanation} />
              <SignalBar label="Tx Frequency"          score={risk.breakdown.frequencyScore} weight="30%" explanation={risk.breakdown.frequencyExplanation} />
              {/* Diversity is inverted: display = 100 - diversityScore so full bar = all in CMC top-50 */}
              <SignalBar
                label="CMC Top-50 Coverage"
                score={100 - risk.breakdown.diversityScore}
                weight="30%"
                explanation={risk.breakdown.diversityExplanation}
                invert
              />
            </div>
          )}

          {/* Source */}
          <div className="mt-4 pt-3 border-t" style={{ borderColor: 'var(--border-subtle)' }}>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Powered by{' '}
              <a href="https://github.com/JoelChng/crypto-portfolio-tracker"
                 target="_blank" rel="noreferrer"
                 className="hover:text-white transition-colors" style={{ color: '#3b82f6' }}>
                crypto-portfolio-tracker
              </a>
            </p>
          </div>
        </>
      )}
    </div>
  )
}

function SignalBar({ label, score, weight, explanation, invert }) {
  const colour = invert
    ? (score >= 70 ? '#22c55e' : score >= 45 ? '#eab308' : score >= 25 ? '#f97316' : '#ef4444')
    : (score <= 30 ? '#22c55e' : score <= 55 ? '#eab308' : score <= 75 ? '#f97316' : '#ef4444')
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span style={{ color: 'var(--text-secondary)' }}>
          {label}{' '}
          <span style={{ color: 'var(--text-muted)' }}>({weight})</span>
        </span>
        <span className="font-semibold" style={{ color: colour }}>{Math.round(score)}</span>
      </div>
      <div className="w-full h-1.5 rounded-full" style={{ backgroundColor: 'var(--border-card)' }}>
        <div className="h-1.5 rounded-full transition-all duration-500"
             style={{ width: `${Math.min(score, 100)}%`, backgroundColor: colour }} />
      </div>
      {explanation && (
        <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{explanation}</p>
      )}
    </div>
  )
}
