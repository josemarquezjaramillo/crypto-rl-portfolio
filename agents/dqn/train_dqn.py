"""
Production training script for DQN portfolio agent.

This script loads the best hyperparameters from the Optuna study and trains
a DQN agent with early stopping based on validation performance. The trained
model is saved with comprehensive checkpointing.

Usage:
    # Load best hyperparameters from Optuna and train
    python agents/dqn/train_dqn.py
    
    # Resume from checkpoint
    python agents/dqn/train_dqn.py --resume
    
    # Override hyperparameters manually (for testing)
    python agents/dqn/train_dqn.py --gamma 0.95 --lr 0.0001 --batch-size 64

Key Features:
    - Automatic loading of best Optuna trial
    - Early stopping with patience (validates every 50 episodes)
    - Checkpoint management (best + latest models)
    - Comprehensive logging (training + validation metrics)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import optuna
from datetime import datetime
from dotenv import load_dotenv
import json

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

TRAINING_CONFIG = {
    'max_episodes': 1000,  # Maximum training episodes
    'min_episodes': 200,   # Minimum episodes before early stopping
    'validation_freq': 50,  # Validate every N episodes
    'patience': 7,          # Stop if no improvement for N validations (7 × 50 = 350 episodes)
    'n_val_episodes': 5,    # Episodes per validation window
    'window_length': 100,   # Days per episode window (matches validation)
    'checkpoint_dir': Path('checkpoints/dqn_production'),
    'log_file': Path('logs/train_dqn.csv'),
}

ENV_CONFIG = {
    'cost_rate': 0.001,
    'turnover_cap': 0.30,
    'max_weight_per_asset': 0.35,
    'strict_projection': False,  # Penalty-based learning (agent learns constraints)
    'constraint_penalty': -10.0,
    'terminate_on_violation': False,
}


# ============================================================================
# Database Connection
# ============================================================================

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
        raise


# ============================================================================
# Environment Setup
# ============================================================================

def create_environments():
    """
    Create training and validation environments.
    
    For training, we return the full train_core backend which will be
    sampled using random sliding windows during training.
    
    Returns
    -------
    ds, train_backend, val_env : tuple
        Dataset (for window creation), training backend, and validation environment
    """
    # Load dataset (dev split contains train_core + validation windows)
    ds = load_exported_dataset("dataset_v1", split="dev")
    
    # Training backend (train_core: 1,848 days)
    # We return the backend instead of env for flexible windowing
    train_backend = DatasetBackend(ds, split_tag_filter="train_core")
    
    # Validation environment (all validation windows: 100 days total)
    val_backend = DatasetBackend(
        ds, 
        split_tag_filter=[
            "val_window_val_2018_crash",
            "val_window_val_covid",
            "val_window_val_bull",
            "val_window_val_bear",
            "val_window_val_chop"
        ]
    )
    val_env = PortfolioEnv(
        EnvConfig(
            split="train",
            random_seed=999,  # Fixed seed for validation consistency
            **ENV_CONFIG
        ),
        val_backend
    )
    
    return ds, train_backend, val_env


# ============================================================================
# Hyperparameter Loading
# ============================================================================

def load_best_hyperparameters(study_name: str = "dqn_portfolio_optimization") -> dict:
    """
    Load best hyperparameters from Optuna study.
    
    Parameters
    ----------
    study_name : str
        Name of Optuna study
        
    Returns
    -------
    params : dict
        Best hyperparameters
    """
    storage_url = get_optuna_storage_url()
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError:
        raise ValueError(f"Study '{study_name}' not found. Run hyperparameter_search.py first.")
    
    if len(study.trials) == 0:
        raise ValueError(f"Study '{study_name}' has no completed trials.")
    
    best_trial = study.best_trial
    
    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS FROM OPTUNA")
    print("="*70)
    print(f"Study: {study_name}")
    print(f"Best Trial: #{best_trial.number}")
    print(f"Best Validation Return: {best_trial.value:.6f}")
    print(f"\nHyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key:25s}: {value}")
    print("="*70 + "\n")
    
    return best_trial.params


def parse_hidden_dims(hidden_dims_str: str) -> list:
    """Convert hidden_dims string to list (e.g., '256_128' -> [256, 128])."""
    hidden_dims_map = {
        '256_128': [256, 128],
        '512_256': [512, 256],
        '1024_512': [1024, 512]
    }
    return hidden_dims_map.get(hidden_dims_str, [256, 128])


# ============================================================================
# Training Loop with Early Stopping
# ============================================================================

def train_with_early_stopping(
    agent: DQNAgent,
    ds,  # ExportedDataset for creating windowed backends
    train_backend: DatasetBackend,
    val_env: PortfolioEnv,
    window_length: int = 100,
    resume: bool = False
) -> dict:
    """
    Train DQN agent with validation-based early stopping using sliding windows.
    
    Instead of sequential episodes through the full train_core dataset,
    each episode samples a random window_length-day window from train_core.
    This provides diverse training experiences and reduces training time.
    
    Parameters
    ----------
    agent : DQNAgent
        Agent to train
    ds : ExportedDataset
        Dataset for creating windowed backends
    train_backend : DatasetBackend
        Training dataset backend for window sampling
    val_env : PortfolioEnv
        Validation environment
    window_length : int
        Number of days per training episode window
    resume : bool
        Whether to resume from checkpoint
        
    Returns
    -------
    training_info : dict
        Summary of training run
    """
    # Setup directories
    checkpoint_dir = TRAINING_CONFIG['checkpoint_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir = checkpoint_dir / 'best'
    latest_dir = checkpoint_dir / 'latest'
    
    # Resume from checkpoint if requested
    start_episode = 0
    if resume and (latest_dir / 'dqn_checkpoint.pt').exists():
        print("Resuming from latest checkpoint...")
        agent.load(latest_dir)
        start_episode = agent.episode_count
    
    # Early stopping state
    best_val_return = -np.inf
    episodes_since_improvement = 0
    validation_history = []
    
    # Setup logging
    log_file = TRAINING_CONFIG['log_file']
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV header
    if not resume or not log_file.exists():
        with open(log_file, 'w') as f:
            f.write('episode,train_return,mean_q,epsilon,buffer_size,')
            f.write('val_return,val_std,is_best\n')
    
    # Calculate window sampling parameters
    all_train_dates = train_backend.dates()  # np.datetime64 array
    total_days = len(all_train_dates)
    max_start_day = total_days - window_length
    
    if max_start_day < 0:
        raise ValueError(
            f"Training dataset ({total_days} days) is shorter than "
            f"window_length ({window_length} days)"
        )
    
    print("\n" + "="*70)
    print("PRODUCTION TRAINING WITH SLIDING WINDOWS")
    print("="*70)
    print(f"Training Dataset:  {total_days} days (train_core)")
    print(f"Window Length:     {window_length} days per episode")
    print(f"Possible Windows:  {max_start_day + 1} starting positions")
    print(f"Max Episodes:      {TRAINING_CONFIG['max_episodes']}")
    print(f"Validation Freq:   Every {TRAINING_CONFIG['validation_freq']} episodes")
    print(f"Early Stop Patience: {TRAINING_CONFIG['patience']} validations")
    print(f"Min Episodes:      {TRAINING_CONFIG['min_episodes']}")
    print(f"Checkpoint Dir:    {checkpoint_dir}")
    print("="*70 + "\n")
    
    # Training loop with sliding windows
    for episode in range(start_episode, TRAINING_CONFIG['max_episodes']):
        # Sample a random 100-day window from train_core
        start_idx = np.random.randint(0, max_start_day + 1)
        end_idx = start_idx + window_length - 1  # inclusive
        
        window_start = str(all_train_dates[start_idx])
        window_end = str(all_train_dates[end_idx])
        
        # Create windowed backend using date range filtering
        window_backend = DatasetBackend(
            ds,
            split_tag_filter="train_core",
            start_date=window_start,
            end_date=window_end
        )
        
        # Create environment for this window
        window_env = PortfolioEnv(
            EnvConfig(
                split="train",
                random_seed=None,  # Different seed per episode for diversity
                **ENV_CONFIG
            ),
            window_backend
        )
        
        # Update agent's environment
        agent.env = window_env
        
        # Train one episode on this window
        episode_metrics = agent.train_episode()
        training_metrics = agent.get_training_metrics()
        
        train_return = episode_metrics.total_reward
        mean_q = training_metrics['mean_q_value']
        epsilon = agent.epsilon
        buffer_size = len(agent.replay_buffer)
        
        # Print progress
        print(f"Episode {episode+1:4d}/{TRAINING_CONFIG['max_episodes']}: "
              f"Return={train_return:7.4f}, "
              f"Q={mean_q:7.2f}, "
              f"ε={epsilon:.3f}, "
              f"Buffer={buffer_size:5d}")
        
        # Validation check
        should_validate = (episode + 1) % TRAINING_CONFIG['validation_freq'] == 0
        is_best = False
        val_return = None
        val_std = None
        
        if should_validate:
            print(f"\n{'─'*70}")
            print(f"VALIDATION at Episode {episode+1}")
            print(f"{'─'*70}")
            
            # Run validation
            val_metrics = agent.evaluate_on_env(
                val_env,
                n_episodes=TRAINING_CONFIG['n_val_episodes'],
                deterministic=True
            )
            
            val_return = val_metrics['mean_return']
            val_std = val_metrics['std_return']
            
            print(f"Validation Return: {val_return:.6f} ± {val_std:.6f}")
            
            # Check for improvement
            if val_return > best_val_return:
                improvement = val_return - best_val_return
                best_val_return = val_return
                episodes_since_improvement = 0
                is_best = True
                
                print(f"✓ NEW BEST! (improvement: +{improvement:.6f})")
                print(f"  Saving to {best_dir}")
                agent.save(best_dir)
                
                # Save best hyperparameters
                with open(best_dir / 'best_val_return.txt', 'w') as f:
                    f.write(f"{best_val_return:.6f}\n")
            else:
                episodes_since_improvement += 1
                print(f"No improvement ({episodes_since_improvement}/{TRAINING_CONFIG['patience']})")
            
            validation_history.append({
                'episode': episode + 1,
                'val_return': val_return,
                'val_std': val_std,
                'is_best': is_best
            })
            
            print(f"{'─'*70}\n")
            
            # Early stopping check
            if (episode + 1) >= TRAINING_CONFIG['min_episodes']:
                if episodes_since_improvement >= TRAINING_CONFIG['patience']:
                    print("\n" + "="*70)
                    print("EARLY STOPPING TRIGGERED")
                    print("="*70)
                    print(f"No improvement for {episodes_since_improvement} validations")
                    print(f"Best validation return: {best_val_return:.6f}")
                    print(f"Stopping at episode {episode+1}")
                    print("="*70 + "\n")
                    break
        
        # Save latest checkpoint
        agent.save(latest_dir)
        
        # Log to CSV
        with open(log_file, 'a') as f:
            f.write(f"{episode+1},{train_return:.6f},{mean_q:.4f},{epsilon:.4f},{buffer_size},")
            if val_return is not None:
                f.write(f"{val_return:.6f},{val_std:.6f},{int(is_best)}\n")
            else:
                f.write(",,\n")
    
    # Training complete
    final_episode = episode + 1
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total Episodes:         {final_episode}")
    print(f"Best Validation Return: {best_val_return:.6f}")
    print(f"Best Model Saved:       {best_dir}")
    print(f"Latest Model Saved:     {latest_dir}")
    print(f"Training Log:           {log_file}")
    print("="*70 + "\n")
    
    # Save training summary
    summary = {
        'total_episodes': final_episode,
        'best_val_return': float(best_val_return),
        'final_epsilon': float(agent.epsilon),
        'validation_history': validation_history,
        'early_stopped': final_episode < TRAINING_CONFIG['max_episodes'],
    }
    
    with open(checkpoint_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train DQN agent with best hyperparameters from Optuna"
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from latest checkpoint'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='dqn_portfolio_optimization',
        help='Name of Optuna study to load hyperparameters from'
    )
    parser.add_argument(
        '--double-dqn',
        action='store_true',
        help='Use Double DQN (reduces Q-value overestimation)'
    )
    
    # Manual hyperparameter overrides (for testing)
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--buffer-size', type=int, help='Replay buffer size')
    parser.add_argument('--epsilon-decay', type=int, help='Epsilon decay episodes')
    parser.add_argument('--epsilon-end', type=float, help='Final epsilon')
    parser.add_argument('--target-update-freq', type=int, help='Target network update frequency')
    parser.add_argument('--hidden-dims', type=str, help='Hidden dimensions (e.g., 256_128)')
    
    args = parser.parse_args()
    
    # Load hyperparameters
    if any([args.gamma, args.lr, args.batch_size, args.buffer_size,
            args.epsilon_decay, args.epsilon_end, args.target_update_freq,
            args.hidden_dims]):
        # Manual override mode
        print("\n⚠️  Using manually specified hyperparameters (not from Optuna)")
        
        # Load defaults from Optuna if not all specified
        try:
            optuna_params = load_best_hyperparameters(args.study_name)
        except:
            # Use reasonable defaults if Optuna not available
            optuna_params = {
                'gamma': 0.95,
                'learning_rate': 0.0001,
                'batch_size': 64,
                'buffer_size': 10000,
                'epsilon_decay_episodes': 500,
                'epsilon_end': 0.05,
                'target_update_freq': 100,
                'hidden_dims': '256_128'
            }
        
        # Override with CLI args
        params = {
            'gamma': args.gamma or optuna_params['gamma'],
            'learning_rate': args.lr or optuna_params['learning_rate'],
            'batch_size': args.batch_size or optuna_params['batch_size'],
            'buffer_size': args.buffer_size or optuna_params['buffer_size'],
            'epsilon_decay_episodes': args.epsilon_decay or optuna_params['epsilon_decay_episodes'],
            'epsilon_end': args.epsilon_end or optuna_params['epsilon_end'],
            'target_update_freq': args.target_update_freq or optuna_params['target_update_freq'],
            'hidden_dims': args.hidden_dims or optuna_params['hidden_dims']
        }
    else:
        # Load from Optuna
        params = load_best_hyperparameters(args.study_name)
    
    # Create environments
    print("Loading dataset and creating environments...")
    ds, train_backend, val_env = create_environments()
    print(f"✓ Training dataset: {len(train_backend.dates())} days (train_core)")
    print(f"✓ Window length: {TRAINING_CONFIG['window_length']} days per episode")
    print(f"✓ Validation environment: {len(val_env.ds.dates())} days")
    
    # Parse hidden dimensions
    hidden_dims = parse_hidden_dims(params['hidden_dims'])
    
    # Determine agent name and checkpoint directory
    agent_name = "ddqn_production" if args.double_dqn else "dqn_production"
    checkpoint_dir = Path(f'checkpoints/{agent_name}')
    
    # Update checkpoint directory and log file in config
    TRAINING_CONFIG['checkpoint_dir'] = checkpoint_dir
    TRAINING_CONFIG['log_file'] = Path(f'logs/train_{agent_name}.csv')
    
    # Create agent configuration
    min_buffer_size = max(params['batch_size'], 100)
    
    agent_config = DQNConfig(
        name=agent_name,
        gamma=params['gamma'],
        learning_rate=params['learning_rate'],
        batch_size=params['batch_size'],
        buffer_size=params['buffer_size'],
        epsilon_start=1.0,
        epsilon_end=params['epsilon_end'],
        epsilon_decay_episodes=params['epsilon_decay_episodes'],
        target_update_freq=params['target_update_freq'],
        hidden_dims=hidden_dims,
        min_buffer_size=min_buffer_size,
        use_double_dqn=args.double_dqn,
        random_seed=42,
        dataset_path="dataset_v1",
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir=None,
        state_dim=256,
        dropout=0.0,
    )
    
    # Print device info
    device = agent_config.device
    print(f"\n✓ Using device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_memory:.2f} GB)")
    
    # Create agent
    agent_type = "Double DQN" if args.double_dqn else "DQN"
    print(f"\nInitializing {agent_type} agent...")
    
    # Create temporary environment for agent initialization
    temp_env = PortfolioEnv(
        EnvConfig(split="train", random_seed=42, **ENV_CONFIG),
        train_backend
    )
    agent = DQNAgent(agent_config, temp_env)
    
    # Train with early stopping using windowed sampling
    summary = train_with_early_stopping(
        agent,
        ds,
        train_backend,
        val_env,
        window_length=TRAINING_CONFIG['window_length'],
        resume=args.resume
    )
    
    print("\n✓ Training complete!")
    print(f"  Best model: {TRAINING_CONFIG['checkpoint_dir'] / 'best'}")
    print(f"  Training log: {TRAINING_CONFIG['log_file']}")


if __name__ == "__main__":
    main()
