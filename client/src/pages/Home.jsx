import WalletInput from '../components/WalletInput'

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4"
         style={{ backgroundColor: '#0f1117' }}>
      <div className="text-center mb-10 max-w-xl">
        <h1 className="text-4xl font-bold text-white mb-3">
          Crypto Risk Profiler
        </h1>
        <p className="text-slate-400 text-lg">
          Paste your Ethereum wallet address to get a personalised risk score
          and portfolio breakdown — free, on-chain, and explainable.
        </p>
      </div>
      <WalletInput />
      <p className="mt-6 text-xs text-slate-600">
        Read-only · No private keys required · Ethereum mainnet
      </p>
    </div>
  )
}
