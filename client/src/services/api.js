import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001'

const api = axios.create({ baseURL: BASE })

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
