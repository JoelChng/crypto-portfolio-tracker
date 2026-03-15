/**
 * Rule-based risk scoring engine.
 * Composite score = 0.4 × holdingsScore + 0.3 × frequencyScore + 0.3 × diversityScore
 * Range: 0–100
 */

function scoreHoldings(tokens) {
  const totalValue = tokens.reduce((s, t) => s + (t.usdValue ?? 0), 0)
  if (totalValue === 0) return { score: 50, explanation: 'No token value data available.' }

  const weightedSum = tokens.reduce((s, t) => {
    const weight = (t.usdValue ?? 0) / totalValue
    return s + weight * (t.riskLevel ?? 2)
  }, 0)

  // weightedSum in [0, 3] → normalise to [0, 100]
  const score = (weightedSum / 3) * 100

  const stablePct = tokens
    .filter(t => t.riskLevel === 0)
    .reduce((s, t) => s + (t.usdValue ?? 0), 0) / totalValue * 100

  const explanation =
    stablePct >= 50
      ? `${stablePct.toFixed(0)}% of your portfolio is in stablecoins, significantly reducing risk.`
      : `Your portfolio is concentrated in higher-risk tokens (weighted average risk tier: ${weightedSum.toFixed(2)}/3).`

  return { score, explanation }
}

function scoreFrequency(transactions, days = 90) {
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000
  const recent = transactions.filter(tx => new Date(tx.date).getTime() >= cutoff)
  const count = recent.length

  let score
  if      (count < 5)   score = 10
  else if (count < 20)  score = 40
  else if (count < 50)  score = 65
  else if (count < 100) score = 80
  else                  score = 100

  const explanation =
    `You made ${count} transaction${count !== 1 ? 's' : ''} in the last 90 days. ` +
    (count < 5
      ? 'Low activity suggests a long-term holding strategy.'
      : count < 20
        ? 'Moderate activity.'
        : 'High frequency trading increases exposure to market volatility.')

  return { score, explanation }
}

function scoreDiversity(tokens) {
  const totalValue = tokens.reduce((s, t) => s + (t.usdValue ?? 0), 0)
  if (totalValue === 0) return { score: 50, explanation: 'No token value data available.' }

  // Tokens outside top-50 (rank > 50 or null = outside top-100)
  const outsideTop50Value = tokens
    .filter(t => t.riskLevel === 3 || (t.rank != null && t.rank > 50))
    .reduce((s, t) => s + (t.usdValue ?? 0), 0)

  const ratio = outsideTop50Value / totalValue
  const score = ratio * 100

  const pct = (ratio * 100).toFixed(0)
  const explanation =
    ratio === 0
      ? 'All holdings are in top-50 tokens by market cap.'
      : `${pct}% of your portfolio is in tokens outside the top 50 by market cap, ` +
        (ratio > 0.5 ? 'which significantly increases your risk exposure.' : 'adding moderate risk.')

  return { score, explanation }
}

function calcRiskScore(tokens, transactions) {
  const holdings  = scoreHoldings(tokens)
  const frequency = scoreFrequency(transactions)
  const diversity = scoreDiversity(tokens)

  const compositeScore =
    0.4 * holdings.score +
    0.3 * frequency.score +
    0.3 * diversity.score

  let category
  if      (compositeScore <= 30) category = 'Conservative 🟢'
  else if (compositeScore <= 55) category = 'Moderate 🟡'
  else if (compositeScore <= 75) category = 'Aggressive 🟠'
  else                           category = 'Very High Risk 🔴'

  return {
    compositeScore,
    category,
    breakdown: {
      holdingsScore:        holdings.score,
      frequencyScore:       frequency.score,
      diversityScore:       diversity.score,
      holdingsExplanation:  holdings.explanation,
      frequencyExplanation: frequency.explanation,
      diversityExplanation: diversity.explanation,
    },
  }
}

module.exports = { calcRiskScore }
