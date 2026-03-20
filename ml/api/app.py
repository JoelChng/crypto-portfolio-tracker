"""
FastAPI credit scoring API.

Endpoints:
  GET  /health
  POST /score          — single wallet score
  POST /batch_score    — up to 100 wallets
  GET  /explain/{wallet} — detailed SHAP explanation
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging
import yaml
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from api.schemas import CreditScoreResponse, BatchScoreRequest, HealthResponse
from api.scoring import CreditScorer

logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/pipeline.yaml"

scorer: CreditScorer = None
reason_codes: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer, reason_codes
    logging.basicConfig(level=logging.INFO)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    with open("configs/reason_codes.yaml") as f:
        reason_codes = yaml.safe_load(f)

    scorer = CreditScorer(
        artifacts_dir=cfg["data"]["artifacts_dir"],
        config=cfg,
        reason_codes=reason_codes,
    )
    try:
        scorer.load()
        logger.info("Models loaded successfully")
    except RuntimeError as e:
        logger.warning(f"Could not load models: {e}. Run training first.")

    yield


app = FastAPI(
    title="On-Chain Credit Scoring API",
    description="ML-powered credit scoring for Ethereum wallets",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoreRequest(BaseModel):
    wallet_address: str
    events: list[dict] = []          # on-chain event records
    horizon_days: int = 90
    explain: bool = True


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = list(scorer._models.keys()) if scorer else []
    return HealthResponse(status="ok", models_loaded=loaded, version="1.0.0")


@app.post("/score", response_model=CreditScoreResponse)
def score_wallet(req: ScoreRequest):
    if not scorer or not scorer._models:
        raise HTTPException(503, "Models not loaded. Run training pipeline first.")

    try:
        empty_cols = ["timestamp","event_type","token","usd_amount",
                      "protocol","gas_fee_usd","health_factor","debt_after_usd"]
        events_df = pd.DataFrame(req.events) if req.events else pd.DataFrame(columns=empty_cols)
        events_df["wallet_address"] = req.wallet_address
        result = scorer.score_wallet(
            events_df=events_df,
            wallet_address=req.wallet_address,
            explain=req.explain,
        )
        return result
    except Exception as e:
        logger.exception(f"Scoring error for {req.wallet_address}")
        raise HTTPException(500, str(e))


@app.post("/batch_score", response_model=list[CreditScoreResponse])
def batch_score(req: BatchScoreRequest):
    if not scorer or not scorer._models:
        raise HTTPException(503, "Models not loaded.")
    results = []
    for wallet in req.wallets[:100]:
        try:
            # Batch requires pre-fetched event data; here we return stub
            raise HTTPException(400, "Batch endpoint requires events data per wallet. "
                                "Use /score endpoint with events payload for each wallet.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch error for {wallet}: {e}")
    return results


if __name__ == "__main__":
    import uvicorn
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    host = cfg.get("api", {}).get("host", "0.0.0.0")
    port = cfg.get("api", {}).get("port", 8000)
    uvicorn.run("api.app:app", host=host, port=port, reload=False)
