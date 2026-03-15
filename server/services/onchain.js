const axios = require('axios')
const NodeCache = require('node-cache')

const cache = new NodeCache({ stdTTL: 300 }) // 5-minute TTL

const MORALIS_BASE = 'https://deep-index.moralis.io/api/v2.2'
const COVALENT_BASE = 'https://api.covalenthq.com/v1'

function moralisHeaders() {
  return { 'X-API-Key': process.env.MORALIS_API_KEY }
}

// ---------- Balances ----------

async function getTokenBalancesMoralis(address) {
  const key = `balances:${address}`
  if (cache.has(key)) return cache.get(key)

  const [erc20Res, ethRes] = await Promise.all([
    axios.get(`${MORALIS_BASE}/${address}/erc20`, {
      headers: moralisHeaders(),
      params: { chain: 'eth' },
    }),
    axios.get(`${MORALIS_BASE}/${address}/balance`, {
      headers: moralisHeaders(),
      params: { chain: 'eth' },
    }),
  ])

  const tokens = erc20Res.data.map(t => ({
    symbol:       t.symbol,
    name:         t.name,
    tokenAddress: t.token_address,
    logo:         t.logo ?? t.thumbnail ?? null,
    balance:      Number(t.balance) / Math.pow(10, Number(t.decimals)),
    decimals:     Number(t.decimals),
  }))

  // Add native ETH
  const ethBalance = Number(ethRes.data.balance) / 1e18
  if (ethBalance > 0) {
    tokens.unshift({
      symbol:       'ETH',
      name:         'Ethereum',
      tokenAddress: 'native',
      logo:         null,
      balance:      ethBalance,
      decimals:     18,
    })
  }

  cache.set(key, tokens)
  return tokens
}

async function getTokenBalancesCovalent(address) {
  const key = `balances_cov:${address}`
  if (cache.has(key)) return cache.get(key)

  const res = await axios.get(
    `${COVALENT_BASE}/eth-mainnet/address/${address}/balances_v2/`,
    {
      auth: { username: process.env.COVALENT_API_KEY, password: '' },
    }
  )

  const items = res.data.data.items ?? []
  const tokens = items
    .filter(t => t.balance !== '0')
    .map(t => ({
      symbol:       t.contract_ticker_symbol,
      name:         t.contract_name,
      tokenAddress: t.contract_address,
      logo:         t.logo_url ?? null,
      balance:      Number(t.balance) / Math.pow(10, t.contract_decimals),
      decimals:     t.contract_decimals,
    }))

  cache.set(key, tokens)
  return tokens
}

async function getTokenBalances(address) {
  if (process.env.MORALIS_API_KEY) {
    return getTokenBalancesMoralis(address)
  }
  return getTokenBalancesCovalent(address)
}

// ---------- Transactions ----------

async function getTransactionsMoralis(address) {
  const key = `txns:${address}`
  if (cache.has(key)) return cache.get(key)

  const res = await axios.get(`${MORALIS_BASE}/${address}/verbose`, {
    headers: moralisHeaders(),
    params: { chain: 'eth', limit: 50 },
  })

  const txns = (res.data.result ?? []).map(tx => {
    const isReceive = tx.to_address?.toLowerCase() === address.toLowerCase()
    const isSend    = tx.from_address?.toLowerCase() === address.toLowerCase()
    let type = 'other'
    if (tx.decoded_call?.label?.toLowerCase().includes('swap')) type = 'swap'
    else if (isReceive) type = 'receive'
    else if (isSend)    type = 'send'

    return {
      hash:        tx.hash,
      date:        tx.block_timestamp,
      type,
      tokenSymbol: 'ETH',
      amount:      Number(tx.value) / 1e18,
      usdValue:    null, // Historical USD value not available in free tier
    }
  })

  cache.set(key, txns)
  return txns
}

async function getTransactionsCovalent(address) {
  const key = `txns_cov:${address}`
  if (cache.has(key)) return cache.get(key)

  const res = await axios.get(
    `${COVALENT_BASE}/eth-mainnet/address/${address}/transactions_v3/`,
    {
      auth: { username: process.env.COVALENT_API_KEY, password: '' },
      params: { 'page-size': 50 },
    }
  )

  const items = res.data.data.items ?? []
  const txns = items.map(tx => {
    const isReceive = tx.to_address?.toLowerCase() === address.toLowerCase()
    const isSend    = tx.from_address?.toLowerCase() === address.toLowerCase()
    const type = isReceive ? 'receive' : isSend ? 'send' : 'other'

    return {
      hash:        tx.tx_hash,
      date:        tx.block_signed_at,
      type,
      tokenSymbol: 'ETH',
      amount:      Number(tx.value) / 1e18,
      usdValue:    tx.value_quote ?? null,
    }
  })

  cache.set(key, txns)
  return txns
}

async function getTransactions(address) {
  if (process.env.MORALIS_API_KEY) {
    return getTransactionsMoralis(address)
  }
  return getTransactionsCovalent(address)
}

module.exports = { getTokenBalances, getTransactions }
