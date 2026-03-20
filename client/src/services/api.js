import axios from 'axios'

const BASE    = import.meta.env.VITE_API_URL    || 'http://localhost:3001'
const ML_BASE = import.meta.env.VITE_ML_API_URL || 'http://localhost:8000'

const api   = axios.create({ baseURL: BASE })
const mlApi = axios.create({ baseURL: ML_BASE, timeout: 20000 })

export async function getPortfolio(address) {
  const { data } = await api.get(`/api/portfolio/${address}`)
  return data
}

export async function getRisk(address) {
  const { data } = await api.get(`/api/risk/${address}`)
  return data
}

export async function getTransactions(address) {
  const { data } = await api.get(`/api/transactions/${address}`)
  return data
}

/**
 * Map express transaction types → ML pipeline event_type strings.
 */
function txTypeToEvent(type) {
  const map = { receive: 'transfer_in', send: 'transfer_out', swap: 'swap' }
  return map[type] || 'transfer_in'
}

/**
 * Call the FastAPI ML scoring server (:8000/score).
 * Returns the credit score response or null if the server is offline.
 */
export async function getCreditScore(address, transactions = []) {
  const events = transactions.map(tx => ({
    timestamp:      tx.date || new Date().toISOString(),
    event_type:     txTypeToEvent(tx.type),
    token:          tx.tokenSymbol || 'ETH',
    usd_amount:     tx.usdValue    || 0,
    protocol:       'unknown',
    gas_fee_usd:    0,
    health_factor:  1.5,
    debt_after_usd: 0,
  }))

  try {
    const { data } = await mlApi.post('/score', {
      wallet_address: address,
      events,
      explain: true,
    })
    return data
  } catch {
    // ML server not running — return null so UI can show offline state
    return null
  }
}

/**
 * Ping the ML API — returns true if reachable.
 */
export async function checkMlHealth() {
  try {
    await mlApi.get('/health', { timeout: 3000 })
    return true
  } catch {
    return false
  }
}
