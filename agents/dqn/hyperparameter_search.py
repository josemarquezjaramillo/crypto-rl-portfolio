"""
Optuna hyperparameter search for DQN portfolio agent.

This script uses Bayesian optimization (TPE) to efficiently search the
hyperparameter space, with automatic pruning of unstable configurations
(Q-value explosion). Results are stored in PostgreSQL for persistence.

Usage:
    python agents/dqn/hyperparameter_search.py --n-trials 100

Study results can be viewed:
    1. PostgreSQL: SELECT * FROM optuna.trials ORDER BY value DESC LIMIT 10
    2. Dashboard: optuna-dashboard postgresql://...
    3. Script output: logs/optuna_study_results.csv
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import signal
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.data_loader import DatabaseConfig
from data.dataset_loader import load_exported_dataset
from data.dataset_backend import DatasetBackend
from environment.environment import PortfolioEnv, EnvConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig


# ============================================================================
# Configuration
# ============================================================================

# Hyperparameter search space (8 hyperparameters following Lucarelli)
SEARCH_CONFIG = {
    'gamma': (0.5, 0.99),  # Discount factor range
    'learning_rate': (1e-5, 1e-3),  # Log-uniform range
    'batch_size': [32, 64, 128],  # Categorical
    'buffer_size': [10000, 50000],  # Categorical
    'epsilon_decay_episodes': [300, 500, 1000],  # Exploration decay
    'epsilon_end': [0.01, 0.05, 0.1],  # Final exploration rate
    'target_update_freq': [50, 100, 200],  # Target network update
    'hidden_dims': ['256_128', '512_256', '1024_512'],  # Network capacity
}

# Training configuration
TRAINING_CONFIG = {
    'n_training_episodes': 50,  # Episodes per trial
    'n_val_episodes': 5,  # Validation episodes (one per regime window)
    'q_explosion_threshold': 10000,  # Prune if Q-values exceed this
    'min_buffer_size': 100,  # Start updating after this many transitions
}

# Environment configuration
ENV_CONFIG = {
    'cost_rate': 0.001,
    'turnover_cap': 0.30,
    'max_weight_per_asset': 0.35,
    'strict_projection': False,
    'constraint_penalty': -10.0,
    'terminate_on_violation': False,
}


# ============================================================================
# Database Connection
# ============================================================================

class TimeoutException(Exception):
    """Raised when a trial exceeds its timeout."""
    pass


@contextmanager
def trial_timeout(seconds: int):
    """
    Context manager to timeout a trial after specified seconds.
    
    Parameters
    ----------
    seconds : int
        Maximum seconds before timeout
    """
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Trial exceeded {seconds} second timeout")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def get_optuna_storage_url() -> str:
    """
    Build Optuna storage URL using existing DatabaseConfig.
    Points to 'optuna' schema in PostgreSQL.
    """
    try:
        db_config = DatabaseConfig()
        # Use search_path option to set default schema to 'optuna'
        return (
            f"postgresql://{db_config.user}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
            f"?options=-csearch_path%3Doptuna"
        )
    except Exception as e:
        print(f"⚠️  Could not connect to PostgreSQL: {e}")
        print(f"Using SQLite storage instead")
        return "sqlite:///logs/optuna_study.db"


# ============================================================================
# Environment Setup
# ============================================================================

def create_environments(trial_number: int):
    """
    Create training and validation environments.
    
    For hyperparameter tuning, we train and validate on the same validation 
    windows (100 days total) to make the search fast. The best hyperparameters
    will then be used for full production training on train_core.
    
    Parameters
    ----------
    trial_number : int
        Trial number for seed variation
    
    Returns
    -------
    train_env, val_env : tuple
        Training and validation environments (both use validation windows)
    """
    # Load dataset (dev split contains train_core + validation windows)
    ds = load_exported_dataset("dataset_v1", split="dev")
    
    # For hyperparameter search: train on validation windows (100 days total)
    # This is 18× faster than using train_core (1,848 days)
    val_windows = [
        "val_window_val_2018_crash",
        "val_window_val_covid",
        "val_window_val_bull",
        "val_window_val_bear",
        "val_window_val_chop"
    ]
    
    train_backend = DatasetBackend(ds, split_tag_filter=val_windows)
    train_env = PortfolioEnv(
        EnvConfig(
            split="train",
            random_seed=42 + trial_number,  # Vary seed per trial
            **ENV_CONFIG
        ),
        train_backend
    )
    
    # Validation uses same windows but with fixed seed for consistency
    val_backend = DatasetBackend(ds, split_tag_filter=val_windows)
    val_env = PortfolioEnv(
        EnvConfig(
            split="train",
            random_seed=999,  # Fixed seed for validation consistency
            **ENV_CONFIG
        ),
        val_backend
    )
    
    return train_env, val_env


# ============================================================================
# Objective Function
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function: trains DQN agent and returns validation return.
    
    Parameters
    ----------
    trial : optuna.Trial
        Current optimization trial
    
    Returns
    -------
    val_return : float
        Mean validation portfolio return (metric to maximize)
    """
    # Sample hyperparameters
    gamma = trial.suggest_float('gamma', *SEARCH_CONFIG['gamma'])
    learning_rate = trial.suggest_float('learning_rate', *SEARCH_CONFIG['learning_rate'], log=True)
    batch_size = trial.suggest_categorical('batch_size', SEARCH_CONFIG['batch_size'])
    buffer_size = trial.suggest_categorical('buffer_size', SEARCH_CONFIG['buffer_size'])
    epsilon_decay_episodes = trial.suggest_categorical('epsilon_decay_episodes', SEARCH_CONFIG['epsilon_decay_episodes'])
    epsilon_end = trial.suggest_float('epsilon_end', *[min(SEARCH_CONFIG['epsilon_end']), max(SEARCH_CONFIG['epsilon_end'])])
    target_update_freq = trial.suggest_categorical('target_update_freq', SEARCH_CONFIG['target_update_freq'])
    hidden_dims_choice = trial.suggest_categorical('hidden_dims', SEARCH_CONFIG['hidden_dims'])
    
    # Map hidden dims choice to actual list
    hidden_dims_map = {
        '256_128': [256, 128],
        '512_256': [512, 256],
        '1024_512': [1024, 512]
    }
    hidden_dims = hidden_dims_map[hidden_dims_choice]
    
    print(f"\n{'='*70}")
    print(f"Trial {trial.number}")
    print(f"{'='*70}")
    print(f"Hyperparameters:")
    print(f"  gamma:                 {gamma:.4f}")
    print(f"  learning_rate:         {learning_rate:.6f}")
    print(f"  batch_size:            {batch_size}")
    print(f"  buffer_size:           {buffer_size}")
    print(f"  epsilon_decay:         {epsilon_decay_episodes}")
    print(f"  epsilon_end:           {epsilon_end:.3f}")
    print(f"  target_update_freq:    {target_update_freq}")
    print(f"  hidden_dims:           {hidden_dims}")
    
    # Create environments
    train_env, val_env = create_environments(trial.number)
    
    # Create agent configuration
    # Set min_buffer_size to max(batch_size, 100) to avoid sampling errors
    min_buffer_size = max(batch_size, TRAINING_CONFIG['min_buffer_size'])
    
    agent_config = DQNConfig(
        name=f"dqn_trial_{trial.number}",
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        epsilon_start=1.0,  # Always start fully random
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        target_update_freq=target_update_freq,
        hidden_dims=hidden_dims,
        min_buffer_size=min_buffer_size,
        random_seed=42 + trial.number,
        dataset_path="dataset_v1",
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir=None,  # No logging during search
        state_dim=256,
        dropout=0.0,
    )
    
    # Create agent
    agent = DQNAgent(agent_config, train_env)
    
    # Training loop
    n_episodes = TRAINING_CONFIG['n_training_episodes']
    for episode in range(n_episodes):
        # Train one episode
        episode_metrics = agent.train_episode()
        
        # Get metrics
        training_metrics = agent.get_training_metrics()
        mean_q = training_metrics['mean_q_value']
        episode_return = episode_metrics.total_reward
        
        # Report episode return to Optuna (for progress tracking)
        trial.report(episode_return, episode)
        
        # Check for Q-value explosion (prune immediately)
        if mean_q > TRAINING_CONFIG['q_explosion_threshold'] or np.isnan(mean_q):
            print(f"\n⚠️  Trial {trial.number} pruned: Q-values exploded ({mean_q:.2e})")
            raise optuna.TrialPruned()
        
        # Let Optuna's MedianPruner decide based on historical trials
        if trial.should_prune():
            print(f"\n⚠️  Trial {trial.number} pruned: underperforming (episode {episode})")
            raise optuna.TrialPruned()
        
        # Progress update every episode (so you can see it's running)
        print(f"  Ep {episode+1:3d}/{n_episodes}: "
              f"Return={episode_return:7.4f}, "
              f"Q={mean_q:7.2f}, "
              f"ε={agent.epsilon:.3f}, "
              f"Buffer={len(agent.replay_buffer):5d}", flush=True)
    
    # Validation evaluation
    print(f"\nEvaluating on validation set...", flush=True)
    try:
        val_metrics = agent.evaluate_on_env(
            val_env, 
            n_episodes=TRAINING_CONFIG['n_val_episodes'],
            deterministic=True,
            max_steps=5000  # Emergency timeout per validation episode
        )
        
        val_return = val_metrics['mean_return']
        print(f"✓ Trial {trial.number} complete:")
        print(f"  Validation return: {val_return:.6f}")
        print(f"  Std return:        {val_metrics['std_return']:.6f}")
        print(flush=True)
        
        return val_return
    except Exception as e:
        print(f"\n✗ Trial {trial.number} failed during validation: {e}")
        print(flush=True)
        raise


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DQN Hyperparameter Search with Optuna")
    parser.add_argument('--n-trials', type=int, default=100, 
                        help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (None = no limit)')
    parser.add_argument('--study-name', type=str, default='dqn_portfolio_optimization',
                        help='Name of the Optuna study')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (1 = sequential)')
    args = parser.parse_args()
    
    # Get storage URL (PostgreSQL or SQLite fallback)
    storage_url = get_optuna_storage_url()
    if 'postgresql' in storage_url:
        print(f"Optuna storage: PostgreSQL (schema: optuna)")
    else:
        print(f"Optuna storage: {storage_url}")
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction='maximize',  # Maximize validation portfolio return
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=50,   # Wait 50 episodes before pruning
        ),
        load_if_exists=True  # Resume if study exists
    )
    
    print(f"\n{'='*70}")
    print(f"STARTING OPTUNA HYPERPARAMETER SEARCH")
    print(f"{'='*70}")
    print(f"Study name:    {args.study_name}")
    print(f"N trials:      {args.n_trials}")
    print(f"Timeout:       {args.timeout}s" if args.timeout else "Timeout:       No limit")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Training:      {TRAINING_CONFIG['n_training_episodes']} episodes per trial")
    print(f"Validation:    {TRAINING_CONFIG['n_val_episodes']} episodes")
    print(f"Metric:        Mean validation return (maximize)")
    print(f"\n")
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # ========================================================================
    # Results Analysis
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    
    print(f"\nBest trial:")
    print(f"  Trial number:      {study.best_trial.number}")
    print(f"  Validation return: {study.best_value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if key == 'learning_rate':
            print(f"  {key:20s}: {value:.6f}")
        elif isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    # Save results to CSV
    results_dir = Path("logs")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"optuna_study_{timestamp}.csv"
    
    df = study.trials_dataframe()
    df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Top 10 trials
    print(f"\nTop 10 trials:")
    top10_cols = ['number', 'value', 'params_gamma', 'params_learning_rate', 
                  'params_batch_size', 'params_buffer_size']
    top10 = df.nlargest(10, 'value')[top10_cols]
    print(top10.to_string(index=False))
    
    # Hyperparameter importance (if enough trials)
    if len(study.trials) >= 10:
        try:
            print(f"\nHyperparameter importance:")
            importance = optuna.importance.get_param_importances(study)
            for param, imp in importance.items():
                print(f"  {param:20s}: {imp:.3f}")
        except Exception as e:
            print(f"Could not compute importance: {e}")
    
    print(f"\n{'='*70}")
    print(f"View results anytime:")
    if 'postgresql' in storage_url:
        print(f"  1. Dashboard:  optuna-dashboard {storage_url}")
        print(f"  2. PostgreSQL: SELECT * FROM optuna.trials WHERE study_name='{args.study_name}'")
    print(f"  3. CSV file:   {results_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
