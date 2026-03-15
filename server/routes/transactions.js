const express = require('express')
const { getTransactions } = require('../services/onchain')

const router = express.Router()
const ETH_ADDRESS = /^0x[0-9a-fA-F]{40}$/

router.get('/:address', async (req, res) => {
  const { address } = req.params
  if (!ETH_ADDRESS.test(address)) {
    return res.status(400).json({ error: 'Invalid Ethereum address' })
  }

  try {
    const txns = await getTransactions(address)
    res.json(txns)
  } catch (err) {
    console.error('Transactions error:', err.message)
    res.status(502).json({ error: 'Failed to fetch transactions', detail: err.message })
  }
})

module.exports = router
