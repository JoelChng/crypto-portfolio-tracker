const express = require('express')
const { getTokenBalances } = require('../services/onchain')
const { enrichTokens } = require('../services/prices')
const { getDb } = require('../db/database')

const router = express.Router()
const ETH_ADDRESS = /^0x[0-9a-fA-F]{40}$/

router.get('/:address', async (req, res) => {
  const { address } = req.params
  if (!ETH_ADDRESS.test(address)) {
    return res.status(400).json({ error: 'Invalid Ethereum address' })
  }

  try {
    const db = getDb()

    // Upsert wallet record
    db.prepare(`
      INSERT INTO wallets (address, last_fetched)
      VALUES (?, datetime('now'))
      ON CONFLICT(address) DO UPDATE SET last_fetched = datetime('now')
    `).run(address)

    const walletRow = db.prepare('SELECT id FROM wallets WHERE address = ?').get(address)

    // Fetch and enrich tokens
    const rawTokens   = await getTokenBalances(address)
    const enriched    = await enrichTokens(rawTokens)
    const withValue   = enriched.filter(t => t.usdValue > 0)
    const totalValueUsd = withValue.reduce((s, t) => s + t.usdValue, 0)

    // Cache in DB
    db.prepare('DELETE FROM token_balances WHERE wallet_id = ?').run(walletRow.id)
    const insertToken = db.prepare(`
      INSERT INTO token_balances
        (wallet_id, token_symbol, token_name, token_address, balance, usd_value, risk_level)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `)
    for (const t of withValue) {
      insertToken.run(walletRow.id, t.symbol, t.name, t.tokenAddress, t.balance, t.usdValue, t.riskLevel)
    }

    res.json({ address, totalValueUsd, tokens: withValue })
  } catch (err) {
    console.error('Portfolio error:', err.message)
    res.status(502).json({ error: 'Failed to fetch portfolio data', detail: err.message })
  }
})

module.exports = router
