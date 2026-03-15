import { useEffect, useState } from 'react'
import { getPortfolio, getRisk } from '../services/api'
import TokenTable from './TokenTable'
import RiskGauge from './RiskGauge'
import TransactionHistory from './TransactionHistory'

export default function Dashboard({ address }) {
  const [portfolio, setPortfolio] = useState(null)
  const [risk, setRisk] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([getPortfolio(address), getRisk(address)])
      .then(([p, r]) => {
        setPortfolio(p)
        setRisk(r)
      })
      .catch(err => setError(err.message || 'Failed to load portfolio'))
      .finally(() => setLoading(false))
  }, [address])

  if (loading) return <LoadingSkeleton />
  if (error) return (
    <div className="rounded-lg p-6 text-center" style={{ backgroundColor: '#1a1d27', border: '1px solid #ef4444' }}>
      <p className="text-red-400 font-medium">Error: {error}</p>
      <p className="text-slate-500 text-sm mt-1">Check the address and try again.</p>
    </div>
  )

  const totalValue = portfolio?.totalValueUsd ?? 0

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard label="Total Value" value={`$${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} />
        <SummaryCard label="Tokens Held" value={portfolio?.tokens?.length ?? 0} />
        <SummaryCard label="Risk Score" value={risk ? `${Math.round(risk.compositeScore)} / 100` : '—'} accent />
      </div>

      {/* Risk gauge + token table */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <TokenTable tokens={portfolio?.tokens ?? []} totalValue={totalValue} />
        </div>
        <div>
          <RiskGauge risk={risk} />
        </div>
      </div>

      {/* Transactions */}
      <TransactionHistory address={address} />
    </div>
  )
}

function SummaryCard({ label, value, accent }) {
  return (
    <div className="rounded-lg p-5" style={{ backgroundColor: '#1a1d27', border: '1px solid #2a2d3e' }}>
      <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-bold" style={{ color: accent ? '#4f8ef7' : '#f1f5f9' }}>{value}</p>
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="grid grid-cols-3 gap-4">
        {[1, 2, 3].map(i => (
          <div key={i} className="h-24 rounded-lg" style={{ backgroundColor: '#1a1d27' }} />
        ))}
      </div>
      <div className="h-64 rounded-lg" style={{ backgroundColor: '#1a1d27' }} />
    </div>
  )
}
