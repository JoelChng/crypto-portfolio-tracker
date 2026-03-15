import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

const ETH_ADDRESS_REGEX = /^0x[0-9a-fA-F]{40}$/

export default function WalletInput() {
  const [address, setAddress] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    const saved = localStorage.getItem('lastWallet')
    if (saved) setAddress(saved)
  }, [])

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = address.trim()
    if (!ETH_ADDRESS_REGEX.test(trimmed)) {
      setError('Please enter a valid Ethereum address (0x followed by 40 hex characters).')
      return
    }
    setError('')
    localStorage.setItem('lastWallet', trimmed)
    navigate(`/portfolio/${trimmed}`)
  }

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-lg">
      <div className="flex gap-2">
        <input
          type="text"
          value={address}
          onChange={e => { setAddress(e.target.value); setError('') }}
          placeholder="0x..."
          spellCheck={false}
          className="flex-1 rounded-lg px-4 py-3 font-mono text-sm text-white outline-none focus:ring-2 transition"
          style={{
            backgroundColor: '#1a1d27',
            border: '1px solid #2a2d3e',
            '--tw-ring-color': '#4f8ef7',
          }}
        />
        <button
          type="submit"
          className="px-5 py-3 rounded-lg font-semibold text-sm text-white transition hover:opacity-90 active:scale-95"
          style={{ backgroundColor: '#4f8ef7' }}
        >
          Analyse
        </button>
      </div>
      {error && (
        <p className="mt-2 text-sm" style={{ color: '#ef4444' }}>{error}</p>
      )}
    </form>
  )
}
