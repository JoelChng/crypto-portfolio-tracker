const PALETTE = [
  '#3b82f6', '#8b5cf6', '#14b8a6', '#f59e0b',
  '#ef4444', '#22c55e', '#ec4899', '#f97316',
  '#06b6d4', '#a78bfa',
]

function buildSegments(tokens, totalValue, r, circum) {
  const sorted  = [...tokens].sort((a, b) => b.usdValue - a.usdValue)
  const top7    = sorted.slice(0, 7)
  const rest    = sorted.slice(7)
  const restVal = rest.reduce((s, t) => s + (t.usdValue || 0), 0)

  const items = restVal > 0 ? [...top7, { symbol: 'Other', usdValue: restVal }] : top7

  let offset = 0
  return items.map((item, i) => {
    const pct  = totalValue > 0 ? item.usdValue / totalValue : 0
    const dash = pct * circum
    const seg  = { ...item, pct, dash, offset, color: PALETTE[i % PALETTE.length] }
    offset += dash
    return seg
  })
}

export default function AllocationChart({ tokens, totalValue }) {
  const SIZE   = 180
  const SW     = 24        // stroke width
  const r      = (SIZE - SW) / 2
  const circum = 2 * Math.PI * r
  const cx     = SIZE / 2
  const cy     = SIZE / 2

  if (!tokens || tokens.length === 0) return null

  const segments = buildSegments(tokens, totalValue, r, circum)

  return (
    <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
      <h2 className="font-semibold text-white mb-4 text-sm">Portfolio Allocation</h2>

      {/* Donut */}
      <div className="flex justify-center mb-4">
        <svg width={SIZE} height={SIZE} style={{ transform: 'rotate(-90deg)' }}>
          {/* Background track */}
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border-card)" strokeWidth={SW} />
          {/* Segments */}
          {segments.map((seg, i) => (
            <circle
              key={i}
              cx={cx} cy={cy} r={r}
              fill="none"
              stroke={seg.color}
              strokeWidth={SW}
              strokeDasharray={`${seg.dash} ${circum - seg.dash}`}
              strokeDashoffset={-seg.offset}
              strokeLinecap="butt"
            />
          ))}
        </svg>
      </div>

      {/* Legend */}
      <div className="space-y-1.5">
        {segments.map((seg, i) => (
          <div key={i} className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: seg.color }} />
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{seg.symbol}</span>
            </div>
            <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
              {(seg.pct * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
