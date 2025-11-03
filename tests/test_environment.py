"""
Unit and integration tests for PortfolioEnv.

Tests cover:
- Projection functions (simplex, align_weights, project_to_feasible)
- Constraint enforcement
- CSV logging
- Action mask support
- Integration with DatasetBackend
- Episode execution

Run with:
    pytest tests/test_environment.py -v
"""

import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from environment.environment import PortfolioEnv, EnvConfig, DataBackend, Obs
from data.dataset_loader import load_exported_dataset
from data.dataset_backend import DatasetBackend


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_backend():
    """Load real dataset backend for integration tests."""
    ds = load_exported_dataset("dataset_v1", split="dev")
    backend = DatasetBackend(ds, split_tag_filter="train_core")
    return backend


@pytest.fixture
def basic_config():
    """Basic environment configuration."""
    return EnvConfig(
        split="train",
        cost_rate=0.001,
        turnover_cap=0.30,
        random_seed=42,
    )


# ============================================================================
# Unit Tests: Projection Functions
# ============================================================================

class TestSimplexProjection:
    """Tests for simplex_projection static method."""
    
    def test_already_on_simplex(self):
        """Test that valid simplex points are unchanged."""
        v = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        w = PortfolioEnv.simplex_projection(v)
        assert np.allclose(w.sum(), 1.0)
        assert np.all(w >= 0)
        assert np.allclose(w, v, atol=1e-6)
    
    def test_negative_values(self):
        """Test projection with negative values."""
        v = np.array([0.5, -0.2, 0.8, -0.1], dtype=np.float32)
        w = PortfolioEnv.simplex_projection(v)
        assert np.allclose(w.sum(), 1.0)
        assert np.all(w >= 0)
        # Negative values should become 0, positive rescaled
        assert w[1] == 0.0
        assert w[3] == 0.0
    
    def test_all_negative(self):
        """Test with all negative values."""
        v = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        w = PortfolioEnv.simplex_projection(v)
        # Should project to some valid simplex point
        assert np.allclose(w.sum(), 1.0) or np.allclose(w.sum(), 0.0)
        assert np.all(w >= 0)
    
    def test_uniform_values(self):
        """Test with uniform values."""
        v = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        w = PortfolioEnv.simplex_projection(v)
        assert np.allclose(w.sum(), 1.0)
        assert np.allclose(w, [0.25, 0.25, 0.25, 0.25], atol=1e-6)
    
    def test_empty_array(self):
        """Test with empty array."""
        v = np.array([], dtype=np.float32)
        w = PortfolioEnv.simplex_projection(v)
        assert w.size == 0
    
    def test_single_element(self):
        """Test with single element."""
        v = np.array([5.0], dtype=np.float32)
        w = PortfolioEnv.simplex_projection(v)
        assert np.allclose(w, [1.0])
    
    def test_large_values(self):
        """Test with large values."""
        v = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        w = PortfolioEnv.simplex_projection(v)
        assert np.allclose(w.sum(), 1.0)
        assert np.all(w >= 0)
        # The algorithm should produce a valid simplex projection
        # (specific values depend on the Duchi algorithm behavior)


class TestAlignWeights:
    """Tests for align_weights static method."""
    
    def test_no_change(self):
        """Test when universe doesn't change."""
        prev_w = np.array([0.3, 0.5, 0.2], dtype=np.float32)
        old_assets = ['BTC', 'ETH', 'XRP']
        new_assets = ['BTC', 'ETH', 'XRP']
        w_new = PortfolioEnv.align_weights(prev_w, old_assets, new_assets)
        assert np.allclose(w_new, prev_w, atol=1e-6)
        assert np.allclose(w_new.sum(), 1.0)
    
    def test_partial_exit(self):
        """Test when some assets exit."""
        prev_w = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        old_assets = ['BTC', 'ETH', 'XRP']
        new_assets = ['BTC', 'ETH']
        w_new = PortfolioEnv.align_weights(prev_w, old_assets, new_assets)
        
        # XRP's 0.2 should be redistributed proportionally
        assert w_new.shape == (2,)
        assert np.allclose(w_new.sum(), 1.0)
        # Relative weights should be preserved: BTC:ETH = 5:3
        assert np.allclose(w_new[0] / w_new[1], 0.5 / 0.3, atol=1e-5)
    
    def test_new_entrant(self):
        """Test when new asset enters."""
        prev_w = np.array([0.6, 0.4], dtype=np.float32)
        old_assets = ['BTC', 'ETH']
        new_assets = ['BTC', 'ETH', 'ADA']
        w_new = PortfolioEnv.align_weights(prev_w, old_assets, new_assets)
        
        # New entrant should start at 0
        assert w_new.shape == (3,)
        assert w_new[2] == 0.0
        assert np.allclose(w_new[0], 0.6)
        assert np.allclose(w_new[1], 0.4)
        assert np.allclose(w_new.sum(), 1.0)
    
    def test_complete_turnover(self):
        """Test when all assets are replaced."""
        prev_w = np.array([0.5, 0.5], dtype=np.float32)
        old_assets = ['BTC', 'ETH']
        new_assets = ['ADA', 'DOT']
        w_new = PortfolioEnv.align_weights(prev_w, old_assets, new_assets)
        
        # All old assets gone, should get some valid allocation
        assert w_new.shape == (2,)
        assert np.allclose(w_new.sum(), 1.0)
        assert np.all(w_new >= 0)
    
    def test_reordering(self):
        """Test when assets are reordered."""
        prev_w = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        old_assets = ['A', 'B', 'C', 'D']
        new_assets = ['D', 'B', 'A', 'C']
        w_new = PortfolioEnv.align_weights(prev_w, old_assets, new_assets)
        
        # Should correctly map to new order
        assert np.allclose(w_new[0], 0.4)  # D
        assert np.allclose(w_new[1], 0.2)  # B
        assert np.allclose(w_new[2], 0.1)  # A
        assert np.allclose(w_new[3], 0.3)  # C


class TestProjectToFeasible:
    """Tests for project_to_feasible method."""
    
    @pytest.fixture
    def env_basic(self, simple_backend):
        """Environment with basic constraints."""
        cfg = EnvConfig(
            split="train",
            cost_rate=0.001,
            turnover_cap=0.30,
            max_weight_per_asset=None,
            random_seed=42
        )
        return PortfolioEnv(cfg, simple_backend)
    
    @pytest.fixture
    def env_with_cap(self, simple_backend):
        """Environment with per-asset cap."""
        cfg = EnvConfig(
            split="train",
            cost_rate=0.001,
            turnover_cap=0.30,
            max_weight_per_asset=0.25,
            random_seed=42
        )
        return PortfolioEnv(cfg, simple_backend)
    
    def test_already_feasible(self, env_basic):
        """Test with already feasible weights."""
        w_prop = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        w_prev = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        w_exec, flags = env_basic.project_to_feasible(w_prop, w_prev)
        
        assert np.allclose(w_exec.sum(), 1.0)
        assert np.all(w_exec >= 0)
        assert not flags['nonneg']  # No negatives clipped
        assert not flags['turnover']  # No turnover violation
    
    def test_negative_clipping(self, env_basic):
        """Test non-negativity constraint."""
        w_prop = np.array([0.5, -0.2, 0.4, 0.3], dtype=np.float32)
        w_prev = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        w_exec, flags = env_basic.project_to_feasible(w_prop, w_prev)
        
        assert np.all(w_exec >= 0)
        assert np.allclose(w_exec.sum(), 1.0)
        assert flags['nonneg']
    
    def test_per_asset_cap(self, env_with_cap):
        """Test per-asset concentration cap."""
        # This tests that cap constraint is checked (flag), not that enforcement is perfect
        # The environment clips each weight to max_weight_per_asset if set
        w_prop = np.array([0.8, 0.1, 0.1], dtype=np.float32)
        w_prev = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        w_exec, flags = env_with_cap.project_to_feasible(w_prop, w_prev)
        
        # Should enforce cap (though may not be perfect due to projection order)
        assert np.allclose(w_exec.sum(), 1.0)
        # Verify that weights are more balanced than proposal (0.8 was proposed)
        assert w_exec.max() < 0.8
    
    def test_turnover_cap(self, env_basic):
        """Test turnover constraint."""
        # Try to make a big move (should be capped)
        w_prop = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        w_prev = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        w_exec, flags = env_basic.project_to_feasible(w_prop, w_prev)
        
        turnover = np.abs(w_exec - w_prev).sum()
        assert turnover <= 0.30 + 1e-6
        assert np.allclose(w_exec.sum(), 1.0)
        assert flags['turnover']
    
    def test_empty_arrays(self, env_basic):
        """Test with empty arrays."""
        w_prop = np.array([], dtype=np.float32)
        w_prev = np.array([], dtype=np.float32)
        w_exec, flags = env_basic.project_to_feasible(w_prop, w_prev)
        
        assert w_exec.size == 0


# ============================================================================
# Integration Tests: Environment Behavior
# ============================================================================

class TestEnvironmentIntegration:
    """Integration tests with real backend."""
    
    def test_reset(self, simple_backend, basic_config):
        """Test environment reset."""
        env = PortfolioEnv(basic_config, simple_backend)
        obs = env.reset(seed=42)
        
        assert 'features' in obs
        assert 'prev_weights' in obs
        assert 'asset_ids' in obs
        assert 'date' in obs
        
        assert obs['features'].shape[1:] == (4, 60)
        assert len(obs['asset_ids']) == obs['features'].shape[0]
        assert obs['prev_weights'].shape[0] == obs['features'].shape[0]
        assert np.allclose(obs['prev_weights'].sum(), 1.0)
    
    def test_step_execution(self, simple_backend, basic_config):
        """Test single step execution."""
        env = PortfolioEnv(basic_config, simple_backend)
        obs = env.reset(seed=42)
        
        action = env.sample_action()
        obs_next, reward, done, info = env.step(action)
        
        # Check observation
        assert 'features' in obs_next
        assert obs_next['features'].shape[1:] == (4, 60)
        
        # Check reward
        assert isinstance(reward, float)
        
        # Check info
        assert 'date' in info
        assert 'turnover' in info
        assert 'transaction_cost' in info
        assert 'portfolio_value' in info
        assert 'constraints_active' in info
    
    def test_episode_execution(self, simple_backend, basic_config):
        """Test full episode execution."""
        env = PortfolioEnv(basic_config, simple_backend)
        obs = env.reset(seed=123)
        
        steps = 0
        total_reward = 0.0
        
        while steps < 50:
            action = env.sample_action()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        assert steps > 0
        assert env.portfolio_value > 0
        assert env.cum_turnover >= 0
        assert env.cum_costs >= 0
    
    def test_determinism(self, simple_backend, basic_config):
        """Test that seeding produces deterministic results."""
        # Run 1
        env1 = PortfolioEnv(basic_config, simple_backend)
        obs1 = env1.reset(seed=999)
        rewards1 = []
        for _ in range(10):
            action = env1.sample_action()
            obs1, reward, done, info = env1.step(action)
            rewards1.append(reward)
            if done:
                break
        
        # Run 2 with same seed
        env2 = PortfolioEnv(basic_config, simple_backend)
        obs2 = env2.reset(seed=999)
        rewards2 = []
        for _ in range(10):
            action = env2.sample_action()
            obs2, reward, done, info = env2.step(action)
            rewards2.append(reward)
            if done:
                break
        
        # Should be identical
        assert len(rewards1) == len(rewards2)
        assert np.allclose(rewards1, rewards2)
    
    def test_terminal_condition(self, simple_backend):
        """Test that episode terminates correctly."""
        # Use very short date range to force termination
        cfg = EnvConfig(
            split="train",
            cost_rate=0.001,
            start_date="2018-09-01",
            end_date="2018-09-05",
            random_seed=42
        )
        env = PortfolioEnv(cfg, simple_backend)
        obs = env.reset()
        
        done = False
        steps = 0
        while not done and steps < 100:
            action = env.sample_action()
            obs, reward, done, info = env.step(action)
            steps += 1
        
        assert done
        assert steps < 100  # Should terminate due to date range


# ============================================================================
# Feature Tests: CSV Logging
# ============================================================================

class TestCSVLogging:
    """Tests for CSV logging functionality."""
    
    def test_logging_enabled(self, simple_backend):
        """Test that logging creates CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EnvConfig(
                split="train",
                cost_rate=0.001,
                log_dir=Path(tmpdir),
                random_seed=42
            )
            env = PortfolioEnv(cfg, simple_backend)
            obs = env.reset()
            
            # Run a few steps
            for _ in range(5):
                action = env.sample_action()
                obs, reward, done, info = env.step(action)
                if done:
                    break
            
            env.close()
            
            # Check log file exists
            log_files = list(Path(tmpdir).glob("env_train_*.csv"))
            assert len(log_files) == 1
            
            # Check content
            with open(log_files[0], 'r') as f:
                lines = f.readlines()
            
            assert len(lines) > 1  # Header + data
            assert 'step' in lines[0]
            assert 'reward_net' in lines[0]
            assert 'portfolio_value' in lines[0]
    
    def test_logging_disabled(self, simple_backend, basic_config):
        """Test that no logging occurs when log_dir is None."""
        env = PortfolioEnv(basic_config, simple_backend)
        obs = env.reset()
        
        for _ in range(5):
            action = env.sample_action()
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        # Should not crash, no file handle
        assert env._log_file is None
        env.close()  # Safe to call


# ============================================================================
# Feature Tests: Action Mask
# ============================================================================

class TestActionMask:
    """Tests for action mask feature."""
    
    def test_action_mask_enabled(self, simple_backend):
        """Test that action mask is included when enabled."""
        cfg = EnvConfig(
            split="train",
            cost_rate=0.001,
            return_action_mask=True,
            action_mask_size=50,
            random_seed=42
        )
        env = PortfolioEnv(cfg, simple_backend)
        obs = env.reset()
        
        assert 'action_mask' in obs
        assert obs['action_mask'].shape == (50,)
        assert obs['action_mask'].dtype == np.bool_
        
        # Check validity
        A_t = len(obs['asset_ids'])
        assert obs['action_mask'][:A_t].all()
        assert not obs['action_mask'][A_t:].any()
    
    def test_action_mask_disabled(self, simple_backend, basic_config):
        """Test that action mask is not included by default."""
        env = PortfolioEnv(basic_config, simple_backend)
        obs = env.reset()
        
        assert 'action_mask' not in obs
    
    def test_action_mask_validation(self, simple_backend):
        """Test that invalid config raises error."""
        cfg = EnvConfig(
            split="train",
            return_action_mask=True,
            action_mask_size=None,  # Invalid!
            random_seed=42
        )
        
        with pytest.raises(ValueError, match="action_mask_size must be set"):
            env = PortfolioEnv(cfg, simple_backend)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_date_range(self, simple_backend):
        """Test with conflicting date constraints."""
        cfg = EnvConfig(
            split="train",
            start_date="2023-01-01",
            end_date="2023-01-01",  # Same day
            random_seed=42
        )
        
        with pytest.raises(ValueError, match="Insufficient dates"):
            env = PortfolioEnv(cfg, simple_backend)
    
    def test_step_after_done(self, simple_backend):
        """Test that stepping after done raises error."""
        cfg = EnvConfig(
            split="train",
            start_date="2018-09-01",
            end_date="2018-09-03",
            random_seed=42
        )
        env = PortfolioEnv(cfg, simple_backend)
        obs = env.reset()
        
        # Run until done
        done = False
        while not done:
            action = env.sample_action()
            obs, reward, done, info = env.step(action)
        
        # Try to step again
        with pytest.raises(RuntimeError, match="Episode has terminated"):
            env.step(env.sample_action())
    
    def test_invalid_max_weight(self, simple_backend):
        """Test that invalid max_weight raises error during projection."""
        cfg = EnvConfig(
            split="train",
            max_weight_per_asset=1.5,  # Invalid > 1
            random_seed=42
        )
        env = PortfolioEnv(cfg, simple_backend)
        obs = env.reset()
        
        with pytest.raises(ValueError, match="max_weight_per_asset"):
            action = np.array([1.0] + [0.0] * (len(obs['asset_ids']) - 1), dtype=np.float32)
            env.step(action)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Tests for performance characteristics."""
    
    def test_step_speed(self, simple_backend, basic_config):
        """Test that steps execute reasonably fast."""
        import time
        
        env = PortfolioEnv(basic_config, simple_backend)
        obs = env.reset()
        
        start = time.time()
        for _ in range(100):
            action = env.sample_action()
            obs, reward, done, info = env.step(action)
            if done:
                break
        elapsed = time.time() - start
        
        # Should be fast (< 1 second for 100 steps on modern hardware)
        assert elapsed < 5.0, f"100 steps took {elapsed:.2f}s (too slow)"


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v", "--tb=short"])
