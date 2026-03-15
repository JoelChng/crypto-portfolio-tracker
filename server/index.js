require('dotenv').config()
const express = require('express')
const cors = require('cors')

const portfolioRouter = require('./routes/portfolio')
const riskRouter = require('./routes/risk')
const transactionsRouter = require('./routes/transactions')

const app = express()
const PORT = process.env.PORT ?? 3001

app.use(cors())
app.use(express.json())

app.use('/api/portfolio', portfolioRouter)
app.use('/api/risk', riskRouter)
app.use('/api/transactions', transactionsRouter)

app.get('/health', (_req, res) => res.json({ status: 'ok' }))

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`))
