/**
 * On-chain data via Etherscan API V2 (free tier).
 *
 * Endpoints used:
 *   account/balance       – native ETH balance
 *   account/tokentx       – ERC-20 transfer history (derive token list)
 *   account/tokenbalance  – current balance for a single ERC-20 contract
 *   account/txlist        – normal ETH transactions
 *
 * Note: tokenlist / addresstokenbalance are Pro-only on V2; we derive
 * token holdings from tokentx history + per-token balance lookups instead.
 */

const axios      = require('axios')
const NodeCache  = require('node-cache')

const cache = new NodeCache({ stdTTL: 300 }) // 5-min TTL

const ETHERSCAN_BASE = 'https://api.etherscan.io/v2/api'

/** Attach API key + chainid=1 (Ethereum mainnet) to every request */
function p(extra) {
  return { chainid: 1, apikey: process.env.ETHERSCAN_API_KEY || '', ...extra }
}

/** Safely parse an Etherscan result array (returns [] on NOTOK / errors) */
function safeArray(res) {
  return res.data?.status === '1' && Array.isArray(res.data.result)
    ? res.data.result
    : []
}

// ──────────────────────────────────────────────────
//  Token Balances
// ──────────────────────────────────────────────────

async function getTokenBalances(address) {
  const cacheKey = `balances:${address}`
  if (cache.has(cacheKey)) return cache.get(cacheKey)

  // ETH balance + ERC-20 transfer history in parallel
  const [ethRes, erc20tokens] = await Promise.all([
    axios.get(ETHERSCAN_BASE, {
      params: p({ module: 'account', action: 'balance', address, tag: 'latest' }),
    }),
    getTokensFromTransferHistory(address),
  ])

  const tokens = []

  // ── Native ETH ──
  const ethBal = Number(ethRes.data?.result ?? 0) / 1e18
  if (ethBal > 0.000001) {
    tokens.push({
      symbol:       'ETH',
      name:         'Ethereum',
      tokenAddress: 'native',
      logo:         'https://assets.coingecko.com/coins/images/279/small/ethereum.png',
      balance:      ethBal,
      decimals:     18,
    })
  }

  tokens.push(...erc20tokens)
  cache.set(cacheKey, tokens)
  return tokens
}

/**
 * Derives held ERC-20 tokens from transfer history, then fetches current
 * balance for each unique contract (tokenlist is Pro-only on V2 free tier).
 */
async function getTokensFromTransferHistory(address) {
  const res   = await axios.get(ETHERSCAN_BASE, {
    params: p({ module: 'account', action: 'tokentx', address, page: 1, offset: 200, sort: 'desc' }),
  })
  const txns = safeArray(res)

  // Unique contracts seen in transfers
  const contracts = {}
  for (const tx of txns) {
    if (tx.contractAddress && !contracts[tx.contractAddress]) {
      contracts[tx.contractAddress] = { symbol: tx.tokenSymbol, name: tx.tokenName, decimals: Number(tx.tokenDecimal ?? 18) }
    }
  }

  const tokens = []
  // Rate limit: Etherscan free = 5 req/s; batch with small pauses
  const entries = Object.entries(contracts).slice(0, 20)
  for (const [contractAddress, meta] of entries) {
    try {
      const r = await axios.get(ETHERSCAN_BASE, {
        params: p({ module: 'account', action: 'tokenbalance', contractaddress: contractAddress, address, tag: 'latest' }),
      })
      const raw = Number(r.data?.result ?? 0)
      const bal = raw / Math.pow(10, meta.decimals)
      if (bal > 0) {
        tokens.push({
          symbol:       meta.symbol || 'TOKEN',
          name:         meta.name   || meta.symbol,
          tokenAddress: contractAddress,
          logo:         null,
          balance:      bal,
          decimals:     meta.decimals,
        })
      }
    } catch { /* skip token on error */ }
  }
  return tokens
}

// ──────────────────────────────────────────────────
//  Transaction History
// ──────────────────────────────────────────────────

async function getTransactions(address) {
  const cacheKey = `txns:${address}`
  if (cache.has(cacheKey)) return cache.get(cacheKey)

  // Fetch normal ETH txns + ERC-20 transfers in parallel
  const [normalRes, erc20Res] = await Promise.all([
    axios.get(ETHERSCAN_BASE, {
      params: p({ module: 'account', action: 'txlist', address,
                  startblock: 0, endblock: 99999999,
                  page: 1, offset: 50, sort: 'desc' }),
    }),
    axios.get(ETHERSCAN_BASE, {
      params: p({ module: 'account', action: 'tokentx', address,
                  page: 1, offset: 50, sort: 'desc' }),
    }),
  ])

  const addrLower = address.toLowerCase()

  // ── Normal ETH transfers ──
  const normalTxns = safeArray(normalRes)
    .filter(tx => tx.isError === '0' && Number(tx.value) > 0)
    .map(tx => {
      const isReceive = tx.to?.toLowerCase()   === addrLower
      const isSend    = tx.from?.toLowerCase() === addrLower
      const isSwap    = tx.functionName?.toLowerCase().includes('swap')
      let type = 'other'
      if (isSwap)    type = 'swap'
      else if (isReceive) type = 'receive'
      else if (isSend)    type = 'send'

      return {
        hash:        tx.hash,
        date:        new Date(Number(tx.timeStamp) * 1000).toISOString(),
        type,
        tokenSymbol: 'ETH',
        amount:      Number(tx.value) / 1e18,
        usdValue:    null,
        from:        tx.from,
        to:          tx.to,
      }
    })

  // ── ERC-20 transfers ──
  const erc20Txns = safeArray(erc20Res)
    .map(tx => {
      const isReceive = tx.to?.toLowerCase() === addrLower
      const dec = Number(tx.tokenDecimal ?? 18)
      return {
        hash:        tx.hash,
        date:        new Date(Number(tx.timeStamp) * 1000).toISOString(),
        type:        isReceive ? 'receive' : 'send',
        tokenSymbol: tx.tokenSymbol || 'TOKEN',
        amount:      Number(tx.value) / Math.pow(10, dec),
        usdValue:    null,
        from:        tx.from,
        to:          tx.to,
      }
    })

  // Merge, deduplicate (same hash+symbol), sort newest first, cap at 50
  const seen = new Set()
  const merged = [...normalTxns, ...erc20Txns]
    .filter(tx => {
      const key = `${tx.hash}:${tx.tokenSymbol}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
    .sort((a, b) => new Date(b.date) - new Date(a.date))
    .slice(0, 50)

  cache.set(cacheKey, merged)
  return merged
}

module.exports = { getTokenBalances, getTransactions }
