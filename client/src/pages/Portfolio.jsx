import { useParams, useNavigate } from 'react-router-dom'
import Dashboard from '../components/Dashboard'

export default function Portfolio() {
  const { address } = useParams()
  const navigate = useNavigate()

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#0f1117' }}>
      <header className="border-b px-6 py-4 flex items-center gap-4"
              style={{ borderColor: '#2a2d3e', backgroundColor: '#1a1d27' }}>
        <button
          onClick={() => navigate('/')}
          className="text-slate-400 hover:text-white text-sm transition-colors"
        >
          ← Back
        </button>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-slate-500 mb-0.5">Wallet</p>
          <p className="text-sm font-mono text-slate-300 truncate">{address}</p>
        </div>
      </header>
      <main className="p-6">
        <Dashboard address={address} />
      </main>
    </div>
  )
}
