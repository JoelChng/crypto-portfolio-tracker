import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

const ETH_ADDRESS_REGEX = /^0x[0-9a-fA-F]{40}$/

const DEMO_ADDRESSES = [
  { label: 'Vitalik.eth',  address: '0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045' },
  { label: 'Binance 14',   address: '0x28C6c06298d514Db089934071355E5743bf21d60' },
]

export default function WalletInput() {
  const [address, setAddress] = useState('')
  const [error,   setError]   = useState('')
  const navigate  = useNavigate()

  useEffect(() => {
    const saved = localStorage.getItem('lastWallet')
    if (saved) setAddress(saved)
  }, [])

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = address.trim()
    if (!ETH_ADDRESS_REGEX.test(trimmed)) {
      setError('Enter a valid Ethereum address (0x + 40 hex characters).')
      return
    }
    setError('')
    localStorage.setItem('lastWallet', trimmed)
    navigate(`/portfolio/${trimmed}`)
  }

  function tryDemo(addr) {
    setAddress(addr)
    setError('')
    localStorage.setItem('lastWallet', addr)
    navigate(`/portfolio/${addr}`)
  }

  return (
    <div className="w-full max-w-xl">
      <form onSubmit={handleSubmit}>
        <div className="flex gap-2">
          <input
            type="text"
            value={address}
            onChange={e => { setAddress(e.target.value); setError('') }}
            placeholder="0x..."
            spellCheck={false}
            autoComplete="off"
            className="flex-1 rounded-xl px-4 py-3 font-mono text-sm text-white outline-none transition-all"
            style={{
              backgroundColor: 'var(--bg-card)',
              border: error ? '1px solid #ef4444' : '1px solid var(--border-card)',
              boxShadow: error ? '0 0 0 3px rgba(239,68,68,0.1)' : undefined,
            }}
            onFocus={e => {
              if (!error) e.target.style.border = '1px solid rgba(59,130,246,0.5)'
              e.target.style.boxShadow = '0 0 0 3px rgba(59,130,246,0.08)'
            }}
            onBlur={e => {
              if (!error) e.target.style.border = '1px solid var(--border-card)'
              e.target.style.boxShadow = 'none'
            }}
          />
          <button
            type="submit"
            className="px-5 py-3 rounded-xl font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-95 flex-shrink-0"
            style={{
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              boxShadow: '0 4px 15px rgba(59,130,246,0.25)',
            }}
          >
            Analyse →
          </button>
        </div>
        {error && (
          <p className="mt-2 text-xs" style={{ color: '#ef4444' }}>{error}</p>
        )}
      </form>

      {/* Demo quick-links */}
      <div className="flex items-center gap-3 mt-3">
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Try:</span>
        {DEMO_ADDRESSES.map(d => (
          <button
            key={d.address}
            onClick={() => tryDemo(d.address)}
            className="text-xs px-2.5 py-1 rounded-lg transition-all hover:border-blue-500/40"
            style={{
              backgroundColor: 'rgba(59,130,246,0.06)',
              border: '1px solid rgba(59,130,246,0.15)',
              color: '#3b82f6',
            }}
          >
            {d.label}
          </button>
        ))}
      </div>
    </div>
  )
}
