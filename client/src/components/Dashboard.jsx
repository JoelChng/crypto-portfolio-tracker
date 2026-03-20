import { useEffect, useState } from 'react'
import { getPortfolio, getRisk, getTransactions, getCreditScore } from '../services/api'
import TokenTable       from './TokenTable'
import RiskGauge        from './RiskGauge'
import TransactionHistory from './TransactionHistory'
import AllocationChart  from './AllocationChart'
import CreditScoreCard  from './CreditScoreCard'

export default function Dashboard({ address }) {
  const [portfolio,    setPortfolio]    = useState(null)
  const [risk,         setRisk]         = useState(null)
  const [creditScore,  setCreditScore]  = useState(undefined)   // undefined = loading
  const [loading,      setLoading]      = useState(true)
  const [error,        setError]        = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setCreditScore(undefined)

    Promise.all([getPortfolio(address), getRisk(address)])
      .then(([p, r]) => {
        setPortfolio(p)
        setRisk(r)
        setLoading(false)

        // Fetch transactions then call ML API (non-blocking)
        getTransactions(address)
          .then(txns => getCreditScore(address, txns))
          .then(cs   => setCreditScore(cs))   // null = offline
          .catch(()  => setCreditScore(null))
      })
      .catch(err => {
        setError(err.message || 'Failed to load portfolio')
        setLoading(false)
        setCreditScore(null)
      })
  }, [address])

  if (loading) return <LoadingSkeleton />

  if (error) return (
    <div className="rounded-xl p-6 text-center"
         style={{ backgroundColor: 'var(--bg-card)', border: '1px solid #ef4444' }}>
      <p className="text-red-400 font-medium">Error: {error}</p>
      <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
        Check the address and try again.
      </p>
    </div>
  )

  const totalValue = portfolio?.totalValueUsd ?? 0

  return (
    <div className="space-y-5">

      {/* ── Summary cards ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <SummaryCard label="Total Value"
          value={`$${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} />
        <SummaryCard label="Tokens Held"
          value={portfolio?.tokens?.length ?? 0} />
        <SummaryCard label="Risk Score (0–100)"
          value={risk ? `${Math.round(risk.compositeScore)} / 100` : '—'}
          sub={risk?.category ?? ''}
          color="#8b5cf6" />
        <SummaryCard label="Credit Score (0–1000)"
          value={creditScore ? `${creditScore.score} / 1000` : creditScore === null ? 'Offline' : '…'}
          sub={creditScore?.risk_grade ? `Grade ${creditScore.risk_grade}` : ''}
          color="#14b8a6" />
      </div>

      {/* ── Token table + Allocation chart ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2">
          <TokenTable tokens={portfolio?.tokens ?? []} totalValue={totalValue} />
        </div>
        <AllocationChart tokens={portfolio?.tokens ?? []} totalValue={totalValue} />
      </div>

      {/* ── Risk Gauge + ML Credit Score ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <RiskGauge risk={risk} />
        <CreditScoreCard creditScore={creditScore} loading={creditScore === undefined} />
      </div>

      {/* ── Transactions ── */}
      <TransactionHistory address={address} />

    </div>
  )
}

function SummaryCard({ label, value, sub, color }) {
  return (
    <div className="rounded-xl p-4 card-glow"
         style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
      <p className="text-xs mb-1 uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>{label}</p>
      <p className="text-xl font-bold leading-tight" style={{ color: color ?? 'var(--text-primary)' }}>
        {value}
      </p>
      {sub && (
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{sub}</p>
      )}
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-4 gap-3">
        {[1,2,3,4].map(i => <div key={i} className="h-20 skeleton" />)}
      </div>
      <div className="h-64 skeleton" />
      <div className="grid grid-cols-2 gap-5">
        <div className="h-72 skeleton" />
        <div className="h-72 skeleton" />
      </div>
    </div>
  )
}
