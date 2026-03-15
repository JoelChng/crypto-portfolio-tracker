const RISK_LABELS = ['Stablecoin', 'Large Cap', 'Mid Cap', 'Small Cap']
const RISK_COLOURS = ['#22c55e', '#4f8ef7', '#eab308', '#ef4444']

export default function TokenTable({ tokens, totalValue }) {
  if (!tokens.length) return (
    <div className="rounded-lg p-6 text-center text-slate-500"
         style={{ backgroundColor: '#1a1d27', border: '1px solid #2a2d3e' }}>
      No token balances found.
    </div>
  )

  return (
    <div className="rounded-lg overflow-hidden" style={{ backgroundColor: '#1a1d27', border: '1px solid #2a2d3e' }}>
      <div className="px-5 py-4 border-b" style={{ borderColor: '#2a2d3e' }}>
        <h2 className="font-semibold text-white">Token Holdings</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-slate-500 uppercase tracking-wider border-b" style={{ borderColor: '#2a2d3e' }}>
              <th className="text-left px-5 py-3">Token</th>
              <th className="text-right px-5 py-3">Balance</th>
              <th className="text-right px-5 py-3">Price</th>
              <th className="text-right px-5 py-3">Value (USD)</th>
              <th className="text-right px-5 py-3">Allocation</th>
              <th className="text-right px-5 py-3">Risk Tier</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((t, i) => {
              const allocation = totalValue > 0 ? (t.usdValue / totalValue) * 100 : 0
              return (
                <tr key={i} className="border-b last:border-0 hover:bg-white/5 transition-colors"
                    style={{ borderColor: '#2a2d3e' }}>
                  <td className="px-5 py-3">
                    <div className="flex items-center gap-2">
                      {t.logo && <img src={t.logo} alt="" className="w-5 h-5 rounded-full" />}
                      <span className="font-medium text-white">{t.symbol}</span>
                      <span className="text-slate-500 text-xs">{t.name}</span>
                    </div>
                  </td>
                  <td className="px-5 py-3 text-right font-mono text-slate-300">
                    {Number(t.balance).toLocaleString('en-US', { maximumFractionDigits: 4 })}
                  </td>
                  <td className="px-5 py-3 text-right text-slate-300">
                    {t.priceUsd != null ? `$${Number(t.priceUsd).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })}` : '—'}
                  </td>
                  <td className="px-5 py-3 text-right font-medium text-white">
                    ${Number(t.usdValue).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-5 py-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="w-16 bg-slate-700 rounded-full h-1.5">
                        <div className="h-1.5 rounded-full" style={{ width: `${Math.min(allocation, 100)}%`, backgroundColor: '#4f8ef7' }} />
                      </div>
                      <span className="text-slate-400 w-10 text-right">{allocation.toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-5 py-3 text-right">
                    <span className="text-xs px-2 py-0.5 rounded-full font-medium"
                          style={{ backgroundColor: RISK_COLOURS[t.riskLevel ?? 2] + '22', color: RISK_COLOURS[t.riskLevel ?? 2] }}>
                      {RISK_LABELS[t.riskLevel ?? 2]}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
