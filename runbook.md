# Avellaneda-Stoikov Double-DQN Market Maker Runbook

This project runs from `doubledqn/as_dqn_market_maker`.

---

## 1. Overview & Data Contract

### Target Outcome
Extends the Avellaneda-Stoikov market-making model with a Double-DQN agent tuning risk-aversion ($\gamma$) and skew parameters every 5 seconds on Bybit `BTCUSDT` L2 order-book data, targeting a static-baseline beat on Sharpe/Sortino over 30 held-out days. Includes a standalone delta-neutral funding-rate arbitrage strategy.

### Data and Output Contract
- `data/` holds local CSV files (`mid_prices.csv`, `book_snapshots.csv`, `trades.csv`, `funding.csv`). Ignored by Git; `.gitkeep` tracked.
- `outputs/` holds run outputs (`tier1/`, `baseline/`, `training/`, `models/`, `evaluation/`). Ignored by Git; `.gitkeep` tracked.
- `src/` holds all implementation code.
- `tests/` holds all automated unit tests.

---

## 2. Environment Setup

### Local / Vast Virtual Environment Setup
```bash
cd doubledqn/as_dqn_market_maker
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Configuration (`.env`)
Create or edit `.env` inside `doubledqn/as_dqn_market_maker/`:
```env
PYTHONPATH=.
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=
```

---

## 3. Automated Test Verification

Before running execution tiers, verify system components via pytest:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/
```

Expected output: `50 passed`.

---

## 4. Execution Tiers

### Tier 1 — Smoke Test (1 Trading Day)
One day, fixed derived $\gamma=1e-05$, $\text{skew}=0$. Confirms data ingestion, 5-second cycle replay, fill simulation, spread sanity check, and Step 12 funding arbitrage position evaluation over 3 settlements.

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_tier1.py \
  --mid-prices data/mid_prices.csv \
  --books data/book_snapshots.csv \
  --trades data/trades.csv \
  --funding data/funding.csv \
  --start 2026-07-01T00:00:00Z \
  --end 2026-07-02T00:00:00Z \
  --output outputs/tier1/tier1_summary.json \
  --pnl-output outputs/tier1/tier1_cycle_pnl.csv
```

### Tier 2 — Small Test (1 Trading Week)
Runs Step 5 static baseline (CMA-ES, reduced 20-generation budget), Step 6 Double-DQN training (reduced step budget), Step 7 evaluation over 1 week, and Step 12 funding arbitrage P&L decomposition over ~21 settlements.

### Tier 3 — Full-Scale Benchmark Run
Runs the full training window (2025-05-01 to 2026-06-30) and 30-day held-out test window (2026-07-01 to 2026-07-30).

1. **Static Baseline Optimization (CMA-ES)**:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/optimize_static_baseline.py \
  --mid-prices data/mid_prices.csv \
  --books data/book_snapshots.csv \
  --trades data/trades.csv \
  --funding data/funding.csv \
  --start 2025-05-01T00:00:00Z \
  --end 2026-06-30T23:59:59Z \
  --output outputs/baseline/static_baseline.json \
  --pnl-output outputs/baseline/static_cycle_pnl.csv
```

2. **Adaptive Double-DQN Training (5 Epochs)**:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/train_adaptive_dqn.py \
  --mid-prices data/mid_prices.csv \
  --books data/book_snapshots.csv \
  --trades data/trades.csv \
  --funding data/funding.csv \
  --start 2025-05-01T00:00:00Z \
  --end 2026-06-30T23:59:59Z \
  --epochs 5 \
  --device cuda \
  --weights-output outputs/models/dqn_weights.pt \
  --pnl-output outputs/training/adaptive_training_cycle_pnl.csv \
  --summary-output outputs/training/dqn_training_summary.json
```

3. **Held-Out 30-Day Evaluation**:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/evaluate_adaptive_dqn.py \
  --mid-prices data/mid_prices.csv \
  --books data/book_snapshots.csv \
  --trades data/trades.csv \
  --funding data/funding.csv \
  --weights outputs/models/dqn_weights.pt \
  --start 2026-07-01T00:00:00Z \
  --end 2026-07-30T23:59:59Z \
  --device cuda \
  --pnl-output outputs/evaluation/adaptive_cycle_pnl.csv
```

4. **Full Evaluation Report Generation**:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_eval.py \
  --static-pnl outputs/baseline/static_cycle_pnl.csv \
  --adaptive-pnl outputs/evaluation/adaptive_cycle_pnl.csv \
  --output outputs/evaluation/evaluation_report.json
```

---

## 5. Vast.ai GPU Deployment Guide

For running Double-DQN training and CMA-ES optimization on GPU nodes:

### Provisioning Instance on Vast.ai
1. Rent an instance with an **NVIDIA A10G (24GB)**, **RTX 4090 (24GB)**, or **A100 (40GB/80GB)**.
2. Image: PyTorch 2.1+ with CUDA 12.1 runtime.
3. SSH into instance:

```bash
cd doubledqn/as_dqn_market_maker
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Running GPU-Accelerated Training
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_adaptive_dqn.py \
  --mid-prices data/mid_prices.csv \
  --books data/book_snapshots.csv \
  --trades data/trades.csv \
  --funding data/funding.csv \
  --start 2025-05-01T00:00:00Z \
  --end 2026-06-30T23:59:59Z \
  --epochs 5 \
  --device cuda
```
