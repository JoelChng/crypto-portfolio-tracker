import { useEffect, useState } from 'react'
import { getTransactions } from '../services/api'

const TYPE_COLOURS = {
  send:    '#ef4444',
  receive: '#22c55e',
  swap:    '#4f8ef7',
  other:   '#94a3b8',
}

export default function TransactionHistory({ address }) {
  const [txns, setTxns] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    getTransactions(address)
      .then(setTxns)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [address])

  return (
    <div className="rounded-lg overflow-hidden" style={{ backgroundColor: '#1a1d27', border: '1px solid #2a2d3e' }}>
      <div className="px-5 py-4 border-b" style={{ borderColor: '#2a2d3e' }}>
        <h2 className="font-semibold text-white">Recent Transactions</h2>
      </div>

      {loading && <div className="p-6 text-center text-slate-500 text-sm animate-pulse">Loading transactions…</div>}
      {error  && <div className="p-6 text-center text-red-400 text-sm">{error}</div>}
      {!loading && !error && txns.length === 0 && (
        <div className="p-6 text-center text-slate-500 text-sm">No transactions found.</div>
      )}

      {!loading && txns.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-slate-500 uppercase tracking-wider border-b" style={{ borderColor: '#2a2d3e' }}>
                <th className="text-left px-5 py-3">Date</th>
                <th className="text-left px-5 py-3">Type</th>
                <th className="text-left px-5 py-3">Token</th>
                <th className="text-right px-5 py-3">Amount</th>
                <th className="text-right px-5 py-3">Value (USD)</th>
                <th className="text-right px-5 py-3">Tx</th>
              </tr>
            </thead>
            <tbody>
              {txns.map((tx, i) => (
                <tr key={i} className="border-b last:border-0 hover:bg-white/5 transition-colors"
                    style={{ borderColor: '#2a2d3e' }}>
                  <td className="px-5 py-3 text-slate-400 whitespace-nowrap">
                    {new Date(tx.date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: '2-digit' })}
                  </td>
                  <td className="px-5 py-3">
                    <span className="text-xs px-2 py-0.5 rounded-full font-medium capitalize"
                          style={{ backgroundColor: (TYPE_COLOURS[tx.type] ?? TYPE_COLOURS.other) + '22',
                                   color: TYPE_COLOURS[tx.type] ?? TYPE_COLOURS.other }}>
                      {tx.type}
                    </span>
                  </td>
                  <td className="px-5 py-3 font-medium text-white">{tx.tokenSymbol ?? 'ETH'}</td>
                  <td className="px-5 py-3 text-right font-mono text-slate-300">
                    {tx.amount != null ? Number(tx.amount).toLocaleString('en-US', { maximumFractionDigits: 6 }) : '—'}
                  </td>
                  <td className="px-5 py-3 text-right text-slate-300">
                    {tx.usdValue != null ? `$${Number(tx.usdValue).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'}
                  </td>
                  <td className="px-5 py-3 text-right">
                    {tx.hash && (
                      <a href={`https://etherscan.io/tx/${tx.hash}`} target="_blank" rel="noreferrer"
                         className="text-xs font-mono hover:underline" style={{ color: '#4f8ef7' }}>
                        {tx.hash.slice(0, 8)}…
                      </a>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
