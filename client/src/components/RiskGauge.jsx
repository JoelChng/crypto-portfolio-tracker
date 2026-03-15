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
  const score = risk ? Math.round(risk.compositeScore) : null
  const category = score != null ? getCategory(score) : null

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || score == null) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    const cx = W / 2, cy = H * 0.72
    const R = W * 0.38

    ctx.clearRect(0, 0, W, H)

    // Background arc
    ctx.beginPath()
    ctx.arc(cx, cy, R, Math.PI, 0)
    ctx.lineWidth = 16
    ctx.strokeStyle = '#2a2d3e'
    ctx.lineCap = 'round'
    ctx.stroke()

    // Coloured arc segments
    const segments = [
      { from: 0,  to: 30,  colour: '#22c55e' },
      { from: 30, to: 55,  colour: '#eab308' },
      { from: 55, to: 75,  colour: '#f97316' },
      { from: 75, to: 100, colour: '#ef4444' },
    ]
    segments.forEach(seg => {
      const aStart = Math.PI + (seg.from / 100) * Math.PI
      const aEnd   = Math.PI + (seg.to   / 100) * Math.PI
      ctx.beginPath()
      ctx.arc(cx, cy, R, aStart, aEnd)
      ctx.lineWidth = 16
      ctx.strokeStyle = seg.colour
      ctx.lineCap = 'butt'
      ctx.stroke()
    })

    // Needle
    const angle = Math.PI + (score / 100) * Math.PI
    const nx = cx + (R - 8) * Math.cos(angle)
    const ny = cy + (R - 8) * Math.sin(angle)
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.lineTo(nx, ny)
    ctx.lineWidth = 3
    ctx.strokeStyle = '#ffffff'
    ctx.lineCap = 'round'
    ctx.stroke()

    // Centre dot
    ctx.beginPath()
    ctx.arc(cx, cy, 5, 0, 2 * Math.PI)
    ctx.fillStyle = '#ffffff'
    ctx.fill()
  }, [score])

  return (
    <div className="rounded-lg p-5 h-full" style={{ backgroundColor: '#1a1d27', border: '1px solid #2a2d3e' }}>
      <h2 className="font-semibold text-white mb-4">Risk Profile</h2>

      {score == null ? (
        <p className="text-slate-500 text-sm">Risk data unavailable.</p>
      ) : (
        <>
          <canvas ref={canvasRef} width={260} height={160} className="w-full max-w-[260px] mx-auto block" />

          <div className="text-center mt-1">
            <p className="text-4xl font-bold text-white">{score}</p>
            <p className="text-sm font-medium mt-0.5" style={{ color: category.colour }}>
              {category.emoji} {category.label}
            </p>
          </div>

          {/* Signal breakdown */}
          {risk.breakdown && (
            <div className="mt-5 space-y-3">
              <SignalBar label="Holdings Composition" score={risk.breakdown.holdingsScore} weight="40%" explanation={risk.breakdown.holdingsExplanation} />
              <SignalBar label="Transaction Frequency" score={risk.breakdown.frequencyScore} weight="30%" explanation={risk.breakdown.frequencyExplanation} />
              <SignalBar label="Market Cap Diversity"  score={risk.breakdown.diversityScore} weight="30%" explanation={risk.breakdown.diversityExplanation} />
            </div>
          )}
        </>
      )}
    </div>
  )
}

function SignalBar({ label, score, weight, explanation }) {
  const colour = score <= 30 ? '#22c55e' : score <= 55 ? '#eab308' : score <= 75 ? '#f97316' : '#ef4444'
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label} <span className="text-slate-600">({weight})</span></span>
        <span className="font-medium" style={{ color: colour }}>{Math.round(score)}</span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-1.5">
        <div className="h-1.5 rounded-full transition-all" style={{ width: `${Math.min(score, 100)}%`, backgroundColor: colour }} />
      </div>
      {explanation && <p className="text-xs text-slate-500 mt-1">{explanation}</p>}
    </div>
  )
}
