import WalletInput from '../components/WalletInput'

const FEATURES = [
  {
    icon: '📊',
    title: 'Portfolio Analytics',
    description: 'Token holdings, live USD prices via CoinGecko, and allocation breakdown for any Ethereum wallet.',
    color: '#3b82f6',
    bg: 'rgba(59,130,246,0.08)',
  },
  {
    icon: '⚡',
    title: 'Rule-Based Risk Score',
    description: 'Instant 0–100 risk score from holdings composition, transaction frequency, and market-cap diversity.',
    color: '#8b5cf6',
    bg: 'rgba(139,92,246,0.08)',
  },
  {
    icon: '🤖',
    title: 'ML Credit Score',
    description: '0–1000 machine-learning credit score with grade A–E and SHAP reason codes — powered by Random Forest (AUC 0.807).',
    color: '#14b8a6',
    bg: 'rgba(20,184,166,0.08)',
  },
]

const STATS = [
  { value: '76', label: 'On-chain features' },
  { value: '0.807', label: 'ML model AUC' },
  { value: '4', label: 'Model ensemble' },
  { value: 'Free', label: 'No subscription' },
]

export default function Home() {
  return (
    <div className="min-h-screen bg-grid hero-glow flex flex-col" style={{ backgroundColor: 'var(--bg-primary)' }}>

      {/* Nav */}
      <nav className="border-b px-6 py-4 flex items-center justify-between"
           style={{ borderColor: 'var(--border-card)', backgroundColor: 'rgba(8,11,20,0.8)', backdropFilter: 'blur(12px)', position: 'sticky', top: 0, zIndex: 50 }}>
        <div className="flex items-center gap-2.5">
          <span className="text-xl">🔐</span>
          <span className="font-bold text-white text-sm tracking-tight">CryptoRisk</span>
        </div>
        <div className="flex items-center gap-6 text-xs" style={{ color: 'var(--text-muted)' }}>
          <a href="https://github.com/JoelChng/crypto-portfolio-tracker" target="_blank" rel="noreferrer"
             className="hover:text-white transition-colors flex items-center gap-1.5">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.154-1.11-1.461-1.11-1.461-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.202 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.741 0 .267.18.578.688.48C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
            </svg>
            Portfolio Tracker
          </a>
          <a href="https://github.com/JoelChng/onchain-credit-scoring" target="_blank" rel="noreferrer"
             className="hover:text-white transition-colors flex items-center gap-1.5">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.154-1.11-1.461-1.11-1.461-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.202 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.741 0 .267.18.578.688.48C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
            </svg>
            ML Scoring
          </a>
        </div>
      </nav>

      {/* Hero */}
      <div className="flex-1 flex flex-col items-center justify-center px-4 py-20 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-8"
             style={{ backgroundColor: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.25)', color: '#3b82f6' }}>
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse inline-block" />
          Live on Ethereum Mainnet · Free &amp; Open Source
        </div>

        <h1 className="text-5xl sm:text-6xl font-black mb-5 leading-tight max-w-3xl">
          <span className="text-white">Crypto Portfolio</span>
          <br />
          <span className="gradient-text">Risk Profiler</span>
        </h1>

        <p className="text-lg max-w-xl mb-10 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          Paste any Ethereum wallet address to get live token holdings,
          a rule-based <strong className="text-white">0–100 risk score</strong>, and a
          machine-learning <strong className="text-white">0–1000 credit score</strong> — free, on-chain, explainable.
        </p>

        <WalletInput />

        <p className="mt-5 text-xs" style={{ color: 'var(--text-muted)' }}>
          Read-only · No private keys required · Ethereum ERC-20
        </p>

        {/* Stats row */}
        <div className="flex flex-wrap justify-center gap-8 mt-16">
          {STATS.map(s => (
            <div key={s.label} className="text-center">
              <div className="text-2xl font-black gradient-text">{s.value}</div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature cards */}
      <div className="max-w-5xl mx-auto w-full px-4 pb-20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {FEATURES.map(f => (
            <div key={f.title} className="rounded-xl p-5 card-glow"
                 style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
              <div className="icon-ring mb-4" style={{ backgroundColor: f.bg }}>
                <span style={{ fontSize: 22 }}>{f.icon}</span>
              </div>
              <h3 className="font-semibold text-white mb-2 text-sm">{f.title}</h3>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                {f.description}
              </p>
            </div>
          ))}
        </div>

        {/* Architecture note */}
        <div className="mt-8 rounded-xl p-5 text-xs"
             style={{ backgroundColor: 'rgba(139,92,246,0.05)', border: '1px solid rgba(139,92,246,0.15)' }}>
          <div className="flex items-start gap-3">
            <span className="mt-0.5" style={{ color: '#8b5cf6' }}>⚙️</span>
            <div style={{ color: 'var(--text-secondary)' }}>
              <span className="font-semibold text-white">Architecture: </span>
              React + Vite frontend calls an Express backend (:3001) for live Moralis/CoinGecko data and a FastAPI ML server (:8000) for credit scoring with Random Forest, XGBoost, LightGBM &amp; Logistic Regression ensemble.
              ML scores appear when the Python API is running locally.
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t px-6 py-5 text-center text-xs"
              style={{ borderColor: 'var(--border-card)', color: 'var(--text-muted)' }}>
        Built by{' '}
        <a href="https://github.com/JoelChng" target="_blank" rel="noreferrer"
           className="hover:text-white transition-colors" style={{ color: 'var(--accent)' }}>
          JoelChng
        </a>
        {' '}· Crypto Portfolio Tracker + On-Chain Credit Scoring · FYP 2025–2026
      </footer>
    </div>
  )
}
