const express = require('express')
const { getTokenBalances } = require('../services/onchain')
const { enrichTokens } = require('../services/prices')
const { getTransactions } = require('../services/onchain')
const { calcRiskScore } = require('../services/riskEngine')
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

    const [rawTokens, transactions] = await Promise.all([
      getTokenBalances(address),
      getTransactions(address),
    ])
    const tokens = await enrichTokens(rawTokens)

    const result = calcRiskScore(tokens, transactions)

    // Persist risk score
    const walletRow = db.prepare('SELECT id FROM wallets WHERE address = ?').get(address)
    if (walletRow) {
      db.prepare(`
        INSERT INTO risk_scores
          (wallet_id, composite_score, holdings_score, frequency_score, diversity_score, category)
        VALUES (?, ?, ?, ?, ?, ?)
      `).run(
        walletRow.id,
        result.compositeScore,
        result.breakdown.holdingsScore,
        result.breakdown.frequencyScore,
        result.breakdown.diversityScore,
        result.category,
      )
    }

    res.json(result)
  } catch (err) {
    console.error('Risk error:', err.message)
    res.status(502).json({ error: 'Failed to calculate risk score', detail: err.message })
  }
})

module.exports = router
