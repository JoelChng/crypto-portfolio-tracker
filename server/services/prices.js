const axios = require('axios')
const NodeCache = require('node-cache')

const cache = new NodeCache({ stdTTL: 300 })

const COINGECKO_BASE = 'https://api.coingecko.com/api/v3'

// Top-100 coins by market cap with their rank (fetched once, cached 5 min)
async function getMarketRanks() {
  const key = 'market_ranks'
  if (cache.has(key)) return cache.get(key)

  const res = await axios.get(`${COINGECKO_BASE}/coins/markets`, {
    params: {
      vs_currency: 'usd',
      order: 'market_cap_desc',
      per_page: 100,
      page: 1,
      sparkline: false,
    },
  })

  // Map: symbol (uppercase) → { rank, id, priceUsd }
  const ranks = {}
  res.data.forEach((coin, idx) => {
    ranks[coin.symbol.toUpperCase()] = {
      rank:     idx + 1,
      id:       coin.id,
      priceUsd: coin.current_price,
    }
  })

  // ETH always present
  cache.set(key, ranks)
  return ranks
}

// Enrich token list with prices and risk levels
async function enrichTokens(tokens) {
  const ranks = await getMarketRanks()

  // For tokens not in top-100, try a batch price lookup by contract address
  // (CoinGecko free tier supports this)
  const unknownAddresses = tokens
    .filter(t => t.tokenAddress !== 'native' && !ranks[t.symbol?.toUpperCase()])
    .map(t => t.tokenAddress)
    .filter(Boolean)

  let extraPrices = {}
  if (unknownAddresses.length > 0) {
    try {
      const res = await axios.get(`${COINGECKO_BASE}/simple/token_price/ethereum`, {
        params: {
          contract_addresses: unknownAddresses.slice(0, 30).join(','),
          vs_currencies: 'usd',
        },
      })
      extraPrices = res.data
    } catch {
      // Non-critical — continue without extra prices
    }
  }

  return tokens.map(token => {
    const symbolKey = token.symbol?.toUpperCase()
    const rankEntry = ranks[symbolKey]
    let priceUsd = null
    let rank = null

    if (token.tokenAddress === 'native') {
      // ETH
      priceUsd = ranks['ETH']?.priceUsd ?? null
      rank = ranks['ETH']?.rank ?? 1
    } else if (rankEntry) {
      priceUsd = rankEntry.priceUsd
      rank = rankEntry.rank
    } else if (extraPrices[token.tokenAddress?.toLowerCase()]) {
      priceUsd = extraPrices[token.tokenAddress.toLowerCase()].usd
      rank = null // not in top 100
    }

    const usdValue = priceUsd != null ? token.balance * priceUsd : 0

    // Risk level: 0=stablecoin, 1=large cap (1-10), 2=mid cap (11-100), 3=small cap
    const STABLECOINS = new Set(['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'FRAX', 'LUSD', 'USDP'])
    let riskLevel
    if (STABLECOINS.has(symbolKey)) {
      riskLevel = 0
    } else if (rank != null && rank <= 10) {
      riskLevel = 1
    } else if (rank != null && rank <= 100) {
      riskLevel = 2
    } else {
      riskLevel = 3
    }

    return { ...token, priceUsd, usdValue, riskLevel, rank }
  })
}

module.exports = { enrichTokens, getMarketRanks }
