const axios     = require('axios')
const NodeCache = require('node-cache')

const cache = new NodeCache({ stdTTL: 300 })   // 5-min TTL for prices
const cmcCache = new NodeCache({ stdTTL: 3600 }) // 1-hour TTL for CMC rankings

const COINGECKO_BASE = 'https://api.coingecko.com/api/v3'
const CMC_BASE       = 'https://pro-api.coinmarketcap.com/v1'

// ─────────────────────────────────────────────
//  CoinGecko — prices + top-100 ranks
// ─────────────────────────────────────────────

async function getMarketRanks() {
  const key = 'market_ranks'
  if (cache.has(key)) return cache.get(key)

  const res = await axios.get(`${COINGECKO_BASE}/coins/markets`, {
    params: {
      vs_currency: 'usd',
      order:       'market_cap_desc',
      per_page:    100,
      page:        1,
      sparkline:   false,
    },
  })

  // Map: SYMBOL (uppercase) → { rank, id, priceUsd }
  const ranks = {}
  res.data.forEach((coin, idx) => {
    ranks[coin.symbol.toUpperCase()] = {
      rank:     idx + 1,
      id:       coin.id,
      priceUsd: coin.current_price,
    }
  })

  cache.set(key, ranks)
  return ranks
}

// ─────────────────────────────────────────────
//  CoinMarketCap — top-50 market cap rankings
//  Used exclusively by scoreDiversity()
// ─────────────────────────────────────────────

async function getCMCTop50() {
  const key = 'cmc_top50'
  if (cmcCache.has(key)) return cmcCache.get(key)

  const apiKey = process.env.CMC_API_KEY
  if (!apiKey || apiKey === 'your_coinmarketcap_api_key_here') {
    return null // key not configured — caller will fall back to CoinGecko rank
  }

  try {
    const res = await axios.get(`${CMC_BASE}/cryptocurrency/listings/latest`, {
      headers: { 'X-CMC_PRO_API_KEY': apiKey },
      params: {
        start:   1,
        limit:   50,
        convert: 'USD',
        sort:    'market_cap',
      },
    })

    // Map: SYMBOL (uppercase) → cmcRank (1-50)
    const ranks = {}
    for (const coin of res.data.data) {
      ranks[coin.symbol.toUpperCase()] = coin.cmc_rank
    }

    cmcCache.set(key, ranks)
    return ranks
  } catch (err) {
    console.error('CMC API error:', err.response?.data?.status?.error_message ?? err.message)
    return null // fall back to CoinGecko rank
  }
}

// ─────────────────────────────────────────────
//  Enrich token list with prices, CoinGecko rank,
//  and CMC top-50 rank
// ─────────────────────────────────────────────

async function enrichTokens(tokens) {
  // Fetch CoinGecko ranks (prices) + CMC top-50 in parallel
  const [cgRanks, cmcRanks] = await Promise.all([
    getMarketRanks(),
    getCMCTop50(),
  ])

  // For tokens not in CoinGecko top-100, batch price lookup by contract address
  const unknownAddresses = tokens
    .filter(t => t.tokenAddress !== 'native' && !cgRanks[t.symbol?.toUpperCase()])
    .map(t => t.tokenAddress)
    .filter(Boolean)

  let extraPrices = {}
  if (unknownAddresses.length > 0) {
    try {
      const res = await axios.get(`${COINGECKO_BASE}/simple/token_price/ethereum`, {
        params: {
          contract_addresses: unknownAddresses.slice(0, 30).join(','),
          vs_currencies:      'usd',
        },
      })
      extraPrices = res.data
    } catch {
      // Non-critical — continue without extra prices
    }
  }

  const STABLECOINS = new Set(['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'FRAX', 'LUSD', 'USDP'])

  return tokens.map(token => {
    const symbolKey  = token.symbol?.toUpperCase()
    const cgEntry    = cgRanks[symbolKey]
    let priceUsd = null
    let cgRank   = null

    if (token.tokenAddress === 'native') {
      priceUsd = cgRanks['ETH']?.priceUsd ?? null
      cgRank   = cgRanks['ETH']?.rank ?? 1
    } else if (cgEntry) {
      priceUsd = cgEntry.priceUsd
      cgRank   = cgEntry.rank
    } else if (extraPrices[token.tokenAddress?.toLowerCase()]) {
      priceUsd = extraPrices[token.tokenAddress.toLowerCase()].usd
      cgRank   = null // not in top-100
    }

    const usdValue = priceUsd != null ? token.balance * priceUsd : 0

    // CoinGecko-based risk tier (used by scoreHoldings)
    let riskLevel
    if (STABLECOINS.has(symbolKey)) {
      riskLevel = 0
    } else if (cgRank != null && cgRank <= 10) {
      riskLevel = 1
    } else if (cgRank != null && cgRank <= 100) {
      riskLevel = 2
    } else {
      riskLevel = 3
    }

    // CoinMarketCap rank — used exclusively by scoreDiversity
    // If CMC data available: use CMC rank; otherwise fall back to CoinGecko rank
    const cmcRank = cmcRanks
      ? (cmcRanks[symbolKey] ?? null)   // null = not in CMC top-50
      : cgRank                           // CMC unavailable — fall back to CoinGecko rank

    return { ...token, priceUsd, usdValue, riskLevel, rank: cgRank, cmcRank }
  })
}

module.exports = { enrichTokens, getMarketRanks, getCMCTop50 }
