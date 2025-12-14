# Reinforcement Learning for Cryptocurrency Portfolio Management

Deep reinforcement learning framework for dynamic cryptocurrency portfolio optimization with weekly rebalancing, realistic transaction costs, and rigorous regime-aware evaluation.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Documentation & Research](#documentation--research)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Usage](#usage)
  - [Training Agents](#training-agents)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Evaluation](#evaluation)
- [Algorithms](#algorithms)
  - [RL Agents](#rl-agents)
  - [Baseline Strategies](#baseline-strategies)
- [Environment Details](#environment-details)
- [Results](#results)
- [Testing](#testing)
- [Checkpoints & Trained Models](#checkpoints--trained-models)
- [Authors](#authors)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project implements and compares reinforcement learning approaches for managing a long-only cryptocurrency portfolio. We train agents on historical market data (2018-2023) and evaluate them against classical portfolio optimization baselines under realistic trading constraints including transaction costs, turnover limits, and monthly universe reconstitution.

**Problem Statement:** Can deep RL agents learn profitable portfolio allocation strategies that outperform classical baselines (Equal-Weight, Market-Cap-Weight, Mean-Variance Optimization) in the highly volatile cryptocurrency market?

**Key Innovation:** Unified MDP environment with strict no-look-ahead guarantees, cost-aware reward formulation, and comprehensive regime-aware evaluation framework inspired by Jiang et al. (2017) and Lucarelli & Borrotti (2020).

---

## Key Features

- **Weekly Rebalancing:** Agents make allocation decisions weekly with daily market observations
- **Realistic Transaction Costs:** 0.10% (10 basis points) per trade, modeling Binance spot trading fees
- **Monthly Universe Reconstitution:** Top-10 cryptocurrencies by market cap, frozen monthly to prevent look-ahead bias
- **Cost-Aware Rewards:** Log-return minus proportional trading costs
- **Rigorous Data Splits:**
  - **Training:** 2018-09-01 to 2021-12-31
  - **Validation:** 2022-01-01 to 2022-12-31
  - **Test:** 2023-01-01 to 2025-10-31
- **60-Day Lookback Window:** Per-asset OHLCV features with proper normalization
- **Hyperparameter Optimization:** Automated tuning via Optuna with Bayesian optimization
- **Comprehensive Evaluation:** Returns, Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, turnover analysis, and detailed visualizations

---

## Documentation & Research

This project includes comprehensive academic documentation:

### Technical Paper
**[EN605741_ReinforecementLearning_TermPaper.pdf](documentation/EN605741_ReinforecementLearning_TermPaper.pdf)**
- Complete methodology, experimental design, and results
- Comparative analysis of RL agents vs. classical baselines
- Regime-aware performance evaluation (bear/bull market analysis)
- Discussion of discrete vs. continuous action spaces

### Presentation
**[EN605741_ReinforecementLearning_TermPresentation.pdf](documentation/EN605741_ReinforecementLearning_TermPresentation.pdf)**
- Quick overview of the project
- Key findings and visualizations
- Lessons learned from RL in finance

### Project Specification
**[PROJECT_SPECIFICATION.md](documentation/PROJECT_SPECIFICATION.md)**
- Detailed technical specification of the MDP formulation
- State representation and normalization procedures
- Action space design (discrete catalog vs. continuous)
- Reward engineering and constraint handling
- Dataset export format and reproducibility guarantees

---

## Project Structure

```
crypto-rl-portfolio/
├── README.md                       # This file
├── requirements.txt                # Python dependencies (PyTorch, Optuna, etc.)
├── requirements-data.txt           # Data processing dependencies
├── .env.example                    # Environment variable template
│
├── agents/                         # RL agent implementations
│   ├── base_agent.py              # Abstract base class with episode tracking
│   ├── dqn/                       # Deep Q-Network agents
│   │   ├── dqn_agent.py          # DQN and Double DQN implementation
│   │   ├── networks.py           # Q-network architectures (GRU-based)
│   │   ├── replay_buffer.py      # Experience replay with vectorized sampling
│   │   ├── action_catalog_delta.py  # Discrete action catalog (70 strategies)
│   │   ├── train_dqn.py          # Production training script
│   │   └── hyperparameter_search.py  # Optuna-based hyperparameter tuning
│   └── policy_grad/               # Policy gradient agents
│       ├── policygrad.py         # REINFORCE implementation
│       └── reinforce_plus_baseline.py  # REINFORCE with value function baseline
│
├── baselines/                      # Classical portfolio strategies
│   ├── equal_weight.py            # Uniform 1/N allocation
│   ├── market_cap_weight.py      # Market-cap weighted allocation
│   └── mean_variance.py          # Markowitz mean-variance optimization
│
├── environment/                    # MDP environment
│   ├── environment.py             # PortfolioEnv class (main environment)
│   └── environment_smoke_run.py  # Sanity check script
│
├── data/                          # Data loading and preprocessing
│   ├── data_loader.py            # PostgreSQL database interface
│   ├── data_builder.py           # OHLCV data cleaning and feature engineering
│   ├── data_exporter.py          # Export dataset artifacts for reproducibility
│   ├── dataset_loader.py         # Load exported datasets
│   ├── dataset_backend.py        # Leakage-safe data access layer
│   └── cache/                    # Cached data artifacts
│
├── evaluation/                     # Performance evaluation framework
│   ├── evaluator.py              # Agent evaluation orchestrator
│   ├── metrics.py                # Financial metrics (Sharpe, Sortino, CAGR, etc.)
│   ├── visualizer.py             # Plotting functions (returns, drawdowns, weights)
│   ├── tables.py                 # LaTeX/Markdown table generators
│   └── run_full_evaluation.py    # CLI for running full evaluation suite
│
├── checkpoints/                    # Saved model checkpoints
│   ├── dqn_production/           # DQN best/latest models
│   ├── ddqn_production/          # Double DQN best/latest models
│   ├── reinforce/                # REINFORCE checkpoints
│   └── reinforce_baseline/       # REINFORCE+baseline checkpoints
│
├── results/                        # Evaluation outputs
│   ├── tables/                   # Performance tables (LaTeX/Markdown)
│   └── visualizations/           # Plots (cumulative returns, weights, etc.)
│       ├── test/
│       └── val/
│
├── logs/                          # Training logs and TensorBoard events
├── notebooks/                     # Jupyter notebooks for analysis
├── tests/                         # Unit tests (pytest)
├── documentation/                 # Academic papers and specifications
│   ├── EN605741_ReinforecementLearning_TermPaper.pdf
│   ├── EN605741_ReinforecementLearning_TermPresentation.pdf
│   ├── PROJECT_SPECIFICATION.md
│   ├── figures/                  # Paper figures
│   ├── sections/                 # LaTeX paper sections
│   └── technical_paper/          # LaTeX source files
│
└── reference_papers/              # Key academic references
```

---

## Requirements

### System Requirements
- **Python:** 3.12+
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** Optional but recommended for faster training (CUDA-compatible)
- **Database:** PostgreSQL 12+ (for raw data storage)

### Key Dependencies
- **Deep Learning:** PyTorch 2.0+, torchvision, torchaudio
- **Optimization:** Optuna 4.0+ (Bayesian hyperparameter tuning)
- **Data Processing:** pandas 2.0+, numpy 1.24+, pyarrow
- **Database:** SQLAlchemy 2.0+, psycopg2-binary 2.9+
- **Testing:** pytest 7.4+, pytest-cov 4.1+
- **Utilities:** python-dotenv 1.0+

See [requirements.txt](requirements.txt) and [requirements-data.txt](requirements-data.txt) for complete dependency lists.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/josemarquezjaramillo/crypto-rl-portfolio.git
cd crypto-rl-portfolio
```

### 2. Create Virtual Environment
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install data processing dependencies
pip install -r requirements-data.txt
```

### 4. Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your database credentials
nano .env  # or use your preferred editor
```

Update the following in `.env`:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crypto_portfolio
DB_USER=postgres
DB_PASSWORD=your_password_here
```

### 5. Set Up PostgreSQL Database
```bash
# Create database
createdb crypto_portfolio

# Or using psql
psql -U postgres -c "CREATE DATABASE crypto_portfolio;"
```

### 6. Verify Installation
```bash
# Test imports
python -c "import torch; import pandas; import sqlalchemy; print('All imports successful!')"

# Run tests
pytest tests/
```

---

## Data Setup

### Option 1: Use Pre-Exported Dataset (Recommended)

The repository includes a pre-exported dataset (`data/dataset_v1.zip`) with all features, labels, and metadata:

```bash
# Extract dataset
cd data
unzip dataset_v1.zip -d dataset_v1/
cd ..
```

The exported dataset includes:
- OHLCV features (60-day lookback windows)
- Market cap weights
- Asset eligibility masks
- Train/validation/test split tags
- Metadata JSON files

### Option 2: Build Dataset from Scratch

If you have raw OHLCV data in PostgreSQL:

```bash
# 1. Build dataset from database
python data/data_builder.py

# 2. Export to frozen dataset format
python data/data_exporter.py --output dataset_v1

# 3. Verify export
python -c "from data.dataset_loader import load_exported_dataset; ds = load_exported_dataset('dataset_v1'); print(f'Loaded {len(ds.features)} samples')"
```

### Data Splits
- **Training:** 2018-09-01 to 2021-12-31 (1,218 days)
- **Validation:** 2022-01-01 to 2022-12-31 (365 days)
- **Test:** 2023-01-01 to 2025-10-31 (1,034 days)

Note: First 60 days (2018-07-01 to 2018-08-31) are used as warmup/context only.

---

## Usage

### Training Agents

#### Train DQN Agent
```bash
# Train with best hyperparameters from Optuna
python agents/dqn/train_dqn.py

# Resume from checkpoint
python agents/dqn/train_dqn.py --resume

# Override hyperparameters
python agents/dqn/train_dqn.py --gamma 0.95 --lr 0.0001 --batch-size 64
```

#### Train REINFORCE Agent
```bash
# Standard REINFORCE
python agents/policy_grad/policygrad.py

# REINFORCE with value function baseline (lower variance)
python agents/policy_grad/reinforce_plus_baseline.py
```

**Training Configuration:**
- Episodes: 1000+ with early stopping (patience=20)
- Validation: Every 50 episodes
- Checkpointing: Best validation Sharpe ratio + latest model
- Logging: Real-time metrics to console and files

### Hyperparameter Optimization

Run Bayesian optimization to find best hyperparameters:

```bash
# DQN hyperparameter search (Optuna)
python agents/dqn/hyperparameter_search.py \
    --n-trials 100 \
    --study-name dqn_optimization \
    --storage sqlite:///optuna_studies.db
```

**Search Space:**
- Learning rate: [1e-5, 1e-3] (log scale)
- Gamma (discount factor): [0.90, 0.999]
- Batch size: [32, 128, 256]
- Replay buffer size: [10000, 100000]
- Target network update frequency: [100, 1000]

Results are saved to SQLite database and can be analyzed via Optuna dashboard.

### Evaluation

#### Quick Evaluation (Summary Metrics Only)
```bash
# Evaluate all agents on test set
python evaluation/run_full_evaluation.py --split test

# Evaluate on validation set
python evaluation/run_full_evaluation.py --split val
```

#### Detailed Evaluation (Metrics + Visualizations)
```bash
# Generate full evaluation suite
python evaluation/run_full_evaluation.py --split test --detailed

# Output:
# - results/tables/test_performance.md
# - results/tables/test_performance.tex
# - results/visualizations/test/cumulative_returns.png
# - results/visualizations/test/drawdown.png
# - results/visualizations/test/weight_evolution.png
# - results/visualizations/test/returns_distribution.png
```

**Evaluation Metrics:**
- Total Return (%)
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside deviation)
- Calmar Ratio (return/max drawdown)
- Maximum Drawdown (%)
- Volatility (annualized)
- Average Daily Turnover (%)
- Hit Rate (% of positive return days)

---

## Algorithms

### RL Agents

#### 1. DQN (Deep Q-Network)
**Type:** Value-based
**Action Space:** Discrete (70-strategy catalog)
**Network:** GRU-based state encoder + Q-value head

**Key Features:**
- Experience replay buffer (100K transitions)
- Target network with soft updates (τ=0.001)
- ε-greedy exploration (ε: 1.0 → 0.01 over 500 episodes)
- Action catalog: Pre-defined portfolio allocations (equal-weight, concentrated positions, diversified strategies)

**File:** [agents/dqn/dqn_agent.py](agents/dqn/dqn_agent.py)

#### 2. Double DQN (DDQN)
**Type:** Value-based with reduced overestimation
**Action Space:** Discrete (70-strategy catalog)
**Network:** Same as DQN

**Key Features:**
- Decouples action selection from value estimation
- Reduces Q-value overestimation bias (Mnih et al., 2015)
- Same discrete action catalog as DQN

**File:** [agents/dqn/dqn_agent.py](agents/dqn/dqn_agent.py) (set `use_double_dqn=True`)

#### 3. REINFORCE
**Type:** Policy gradient
**Action Space:** Continuous (softmax over assets)
**Network:** GRU encoder + policy head (outputs asset logits)

**Key Features:**
- Direct policy optimization (no value function)
- Continuous action space (more flexible than discrete catalog)
- Monte Carlo returns (full episode rollouts)
- Baseline variant available for variance reduction

**Files:**
- [agents/policy_grad/policygrad.py](agents/policy_grad/policygrad.py) (vanilla)
- [agents/policy_grad/reinforce_plus_baseline.py](agents/policy_grad/reinforce_plus_baseline.py) (with baseline)

### Baseline Strategies

#### 1. Equal-Weight (1/N)
Allocates equal weight to all tradable assets.

**Formula:** w_i = 1/N where N = number of assets

**Advantages:** Simple, robust, low turnover
**File:** [baselines/equal_weight.py](baselines/equal_weight.py)

#### 2. Market-Cap-Weight
Allocates proportional to market capitalization.

**Formula:** w_i = market_cap_i / Σ market_cap_j

**Advantages:** Market-neutral, captures momentum
**File:** [baselines/market_cap_weight.py](baselines/market_cap_weight.py)

#### 3. Mean-Variance Optimization (Markowitz)
Minimizes portfolio variance subject to return constraints.

**Formulation:**
```
minimize     w^T Σ w
subject to   μ^T w ≥ μ_target
             Σ w_i = 1
             w_i ≥ 0  (long-only)
             w_i ≤ w_max  (concentration limit)
```

**Advantages:** Theoretically optimal for risk-return tradeoff
**File:** [baselines/mean_variance.py](baselines/mean_variance.py)

---

## Environment Details

### MDP Formulation

**State Space:**
- Per-asset features: [A_t, 4, 60] where A_t = number of tradable assets
  - 4 channels: Close, High, Low, Volume
  - 60-day lookback window
- Previous portfolio weights: [A_t]
- Asset IDs: List[str] (for tracking dynamic universe)

**Action Space:**
- **Discrete Mode:** Index into 70-strategy catalog
- **Continuous Mode:** Softmax weights over assets (Σ w_i = 1, w_i ≥ 0)

**Reward:**
```
r_t = log(portfolio_value_t / portfolio_value_{t-1}) - cost_rate × turnover_t
```
where turnover_t = Σ |w_t - w_{t-1}|

**Constraints:**
- Long-only: w_i ≥ 0
- Fully invested: Σ w_i = 1
- Concentration limit: w_i ≤ 0.50 (no single asset > 50%)
- Turnover cap: Σ |w_t - w_{t-1}| ≤ 0.30 (30% daily limit)

**Episode Structure:**
- Start date: First day of train/val/test split
- End date: Last day of split
- Rebalancing frequency: Daily
- Episode length: ~365-1,200 days (varies by split)

### Data Normalization

**Price Features:**
- Divide Close, High, Low by current Close (relative pricing)
- Most recent bar normalized to 1.0

**Volume:**
- log(1 + volume) transformation
- Z-score normalization within 60-day window

**Prevents look-ahead bias:** All normalization uses only data available at decision time t.

---

## Results

### Test Set Performance (2023-01-01 to 2025-10-31)

| Agent | Return (%) | CAGR (%) | Sharpe | Sortino | Max DD (%) | Turnover (%) |
|-------|------------|----------|--------|---------|------------|--------------|
| **Mean-Variance** | **366.5** | **139.1** | **1.64** | **1.69** | **40.0** | 13.5 |
| Equal-Weight | 168.5 | 74.9 | 1.22 | 1.25 | 48.9 | 0.3 |
| Market-Cap-Weight | 159.3 | 71.5 | 1.24 | 1.30 | 44.2 | 0.3 |
| DQN | 155.5 | 70.0 | 1.15 | 1.18 | 52.6 | 10.9 |
| DDQN | 144.5 | 65.9 | 1.14 | 1.16 | 52.6 | 8.4 |

### Key Findings

**RL Underperformance:**
- DQN/DDQN achieve lower returns and Sharpe ratios than all classical baselines
- Mean-Variance optimization dominates with 366% return vs. ~155% for DQN

**Possible Explanations:**
1. **Discrete action catalog limitation:** 70 pre-defined strategies may be too restrictive
2. **Regime shift:** Training period (2018-2023 bear/sideways) differs from test period (2024-2025 bull market)
3. **Sample efficiency:** RL agents may need more training data or episodes
4. **Overfitting to training regime:** Agents optimized for volatility/mean-reversion, not momentum

**Consistent with Literature:**
Lucarelli & Borrotti (2020) found similar RL underperformance in crypto markets, attributing it to regime sensitivity and non-stationarity.

**Detailed results and visualizations:** See [results/](results/) directory and technical paper.

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_environment.py

# Verbose output
pytest -v
```

**Test Coverage:**
- Environment MDP logic (state transitions, rewards, constraints)
- Data loading and preprocessing
- Agent action selection and learning updates
- Baseline strategy allocations
- Metrics calculations

---

## Checkpoints & Trained Models

Pre-trained models are available in the [checkpoints/](checkpoints/) directory:

```
checkpoints/
├── dqn_production/
│   ├── best/           # Best validation Sharpe ratio
│   └── latest/         # Most recent checkpoint
├── ddqn_production/
│   ├── best/
│   └── latest/
├── reinforce/
│   └── best/
└── reinforce_baseline/
    └── best/
```

### Loading Checkpoints

```python
from agents.dqn import DQNAgent, DQNConfig
import torch

# Load DQN agent
config = DQNConfig.from_json("checkpoints/dqn_production/best/config.json")
agent = DQNAgent(config, env)
agent.q_network.load_state_dict(
    torch.load("checkpoints/dqn_production/best/q_network.pt")
)
agent.eval()
```

---

## Authors

**Jose Márquez Jaramillo** & **Taylor Hawks**
Johns Hopkins University
EN605.741 Reinforcement Learning
Fall 2024

**GitHub Repository:** [https://github.com/josemarquezjaramillo/crypto-rl-portfolio](https://github.com/josemarquezjaramillo/crypto-rl-portfolio)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@techreport{marquez2024crypto,
  title={Reinforcement Learning for Cryptocurrency Portfolio Management: A Comparative Study of Value-Based and Policy-Gradient Approaches},
  author={M\'{a}rquez Jaramillo, Jose and Hawks, Taylor},
  institution={Johns Hopkins University},
  year={2024},
  type={Course Project Report},
  note={EN605.741 Reinforcement Learning}
}
```

---

## Acknowledgments

- **Course Instructor:** EN605.741 Reinforcement Learning, Johns Hopkins University
- **Data Sources:** Cryptocurrency market data via public APIs
- **Key References:**
  - Jiang et al. (2017): *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem*
  - Lucarelli & Borrotti (2020): *A Deep Q-Learning Portfolio Management Framework for the Cryptocurrency Market*
  - Mnih et al. (2015): *Human-level control through deep reinforcement learning*
- **Libraries:** PyTorch, Optuna, pandas, NumPy, PostgreSQL, SQLAlchemy

---

**Questions or Issues?** Please open an issue on GitHub or contact the authors.
