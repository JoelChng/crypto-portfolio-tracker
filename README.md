# Crypto Portfolio Tracker

A web-based, on-chain cryptocurrency portfolio tracker with AI-powered risk profiling for retail investors.

## Overview

Paste your Ethereum wallet address to get:
- **Token holdings** with live USD prices via CoinGecko
- **Portfolio allocation** breakdown
- **Risk score (0–100)** based on holdings composition, transaction frequency, and market cap diversity
- **Transaction history** with Etherscan links

> Free alternative to institutional tools like Nansen ($99+/mo).

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite + Tailwind CSS |
| Backend | Node.js + Express |
| On-chain data | Moralis API (Covalent fallback) |
| Live prices | CoinGecko API |
| Database | SQLite |

## Quick Start

```bash
# 1. Install dependencies
cd server && npm install
cd ../client && npm install

# 2. Configure API key
cp server/.env.example server/.env
# Add your MORALIS_API_KEY to server/.env

# 3. Run both servers
cd .. && npm run dev
```

Frontend: http://localhost:5173
Backend:  http://localhost:3001

## Risk Scoring

The risk engine produces a **0–100 score** from three signals:

| Signal | Weight | Description |
|---|---|---|
| Holdings Composition | 40% | Stablecoin vs large/mid/small-cap token split |
| Transaction Frequency | 30% | Activity in last 90 days |
| Market Cap Diversity | 30% | % portfolio outside top-50 tokens |

**Score categories:** Conservative (0–30) · Moderate (31–55) · Aggressive (56–75) · Very High Risk (76–100)

## Project Structure

```
├── client/          # React frontend (Vite)
│   └── src/
│       ├── components/   # Dashboard, RiskGauge, TokenTable, etc.
│       ├── pages/        # Home, Portfolio
│       └── services/     # API client
└── server/          # Express backend
    ├── routes/       # /api/portfolio, /api/risk, /api/transactions
    ├── services/     # onchain.js, prices.js, riskEngine.js
    └── db/           # SQLite schema
```
