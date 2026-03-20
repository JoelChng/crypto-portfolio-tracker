import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import Dashboard from '../components/Dashboard'

function truncateAddress(addr) {
  if (!addr || addr.length < 12) return addr
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`
}

export default function Portfolio() {
  const { address } = useParams()
  const navigate    = useNavigate()
  const [copied, setCopied]   = useState(false)

  function copyAddress() {
    navigator.clipboard.writeText(address).catch(() => {})
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--bg-primary)' }}>

      {/* ── Header ── */}
      <header className="sticky top-0 z-40 border-b px-5 py-3 flex items-center gap-4"
              style={{
                borderColor: 'var(--border-card)',
                backgroundColor: 'rgba(8,11,20,0.9)',
                backdropFilter: 'blur(14px)',
              }}>

        {/* Logo + back */}
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-sm transition-colors group"
          style={{ color: 'var(--text-muted)' }}
        >
          <span className="group-hover:text-white transition-colors">←</span>
          <span className="hidden sm:inline font-bold text-white">🔐 CryptoRisk</span>
        </button>

        <div className="w-px h-5 hidden sm:block" style={{ backgroundColor: 'var(--border-card)' }} />

        {/* Wallet address */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <div className="w-2 h-2 rounded-full bg-green-400 flex-shrink-0 animate-pulse" />
          <span className="text-xs font-mono hidden md:block" style={{ color: 'var(--text-secondary)' }}>
            {address}
          </span>
          <span className="text-xs font-mono md:hidden" style={{ color: 'var(--text-secondary)' }}>
            {truncateAddress(address)}
          </span>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={copyAddress}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all"
            style={{
              backgroundColor: copied ? 'rgba(34,197,94,0.1)' : 'rgba(255,255,255,0.04)',
              border: `1px solid ${copied ? 'rgba(34,197,94,0.3)' : 'var(--border-card)'}`,
              color: copied ? '#22c55e' : 'var(--text-secondary)',
            }}
          >
            {copied ? '✓ Copied' : '⧉ Copy'}
          </button>
          <a
            href={`https://etherscan.io/address/${address}`}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all hover:border-blue-500/40"
            style={{
              backgroundColor: 'rgba(255,255,255,0.04)',
              border: '1px solid var(--border-card)',
              color: 'var(--text-secondary)',
            }}
          >
            Etherscan ↗
          </a>
        </div>
      </header>

      {/* ── Main content ── */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <Dashboard address={address} />
      </main>

      {/* ── Footer ── */}
      <footer className="border-t mt-10 px-6 py-4 text-center text-xs"
              style={{ borderColor: 'var(--border-card)', color: 'var(--text-muted)' }}>
        Data: Moralis · Prices: CoinGecko · ML: onchain-credit-scoring · Read-only
      </footer>
    </div>
  )
}
