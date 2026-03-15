"""
End-to-end pipeline runner.

Usage:
    python run_pipeline.py                        # run all stages
    python run_pipeline.py --stage data           # generate synthetic data only
    python run_pipeline.py --stage features       # feature engineering only
    python run_pipeline.py --stage train          # training only
    python run_pipeline.py --stage explain        # SHAP explainability only
    python run_pipeline.py --stage api            # start API server
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ensure ml/ is on PYTHONPATH regardless of where script is called from
sys.path.insert(0, str(Path(__file__).parent))


def stage_data(config_path: str):
    logger.info("━━━ Stage 1: Generating synthetic data ━━━")
    from data.synthetic_generator import run as gen_run
    from data.snapshot_builder import run as snap_run
    from data.label_generator import run as label_run
    gen_run(config_path)
    snap_run(config_path)
    label_run(config_path)
    logger.info("✓ Data stage complete\n")


def stage_features(config_path: str):
    logger.info("━━━ Stage 2: Feature engineering ━━━")
    from features.assembler import run as feat_run
    feat_run(config_path)
    logger.info("✓ Features stage complete\n")


def stage_train(config_path: str, horizon: int = 90):
    logger.info(f"━━━ Stage 3: Training models (horizon={horizon}d) ━━━")
    from training.train_all import run as train_run
    summary = train_run(config_path, horizon)
    logger.info(f"Champion model: {summary['champion']}")
    logger.info("✓ Training stage complete\n")


def stage_explain(config_path: str):
    logger.info("━━━ Stage 4: SHAP explainability ━━━")
    from explainability.shap_explainer import run_global_shap
    run_global_shap(config_path)
    logger.info("✓ Explainability stage complete\n")


def stage_api(config_path: str):
    logger.info("━━━ Stage 5: Starting API server ━━━")
    import uvicorn
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    host = cfg.get("api", {}).get("host", "0.0.0.0")
    port = cfg.get("api", {}).get("port", 8000)
    uvicorn.run("api.app:app", host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="On-chain credit scoring pipeline")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--stage", default="all",
                        choices=["all", "data", "features", "train", "explain", "api"])
    parser.add_argument("--horizon", type=int, default=90)
    args = parser.parse_args()

    cfg = args.config

    if args.stage in ("all", "data"):
        stage_data(cfg)
    if args.stage in ("all", "features"):
        stage_features(cfg)
    if args.stage in ("all", "train"):
        stage_train(cfg, args.horizon)
    if args.stage in ("all", "explain"):
        stage_explain(cfg)
    if args.stage == "api":
        stage_api(cfg)

    if args.stage == "all":
        logger.info("🎉 Full pipeline complete!")
        logger.info("Run  python run_pipeline.py --stage api  to start the scoring API.")


if __name__ == "__main__":
    main()
