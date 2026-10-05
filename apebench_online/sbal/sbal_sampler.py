"""SBAL (Simulation-Based Active Learning) sampler for APEBench.

This module implements pool-based active learning for selecting simulation parameters
using ensemble uncertainty. Inspired by AL4PDE (https://arxiv.org/abs/2408.01536).

The SBAL approach:
1. At each resampling event, generate a large pool of candidate parameters
2. Use the ensemble model to predict uncertainty for each candidate (via rollout)
3. Select candidates with highest uncertainty for simulation

Usage in config:
    "active_sampling_config": {
        "breeder_backend": "sbal",
        "sbal_config": {
            "pool_size": 5000,
            "selection_strategy": "power",
            "power_beta": 1.0,
            "rollout_steps_rel": 0.5,
        }
    }
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SBALConfig:
    """Configuration for SBAL active learning.
    
    Hyperparameters aligned with AL4PDE defaults for fair comparison:
    - pool_size: 5,000 (reduced from AL4PDE's 100k for efficiency)
    - rollout_steps_rel: 0.5 (50% of trajectory for faster uncertainty computation) 
    - selection_strategy: "power" with beta=1
    """
    # Pool size: reduced from AL4PDE's 100,000 for efficiency
    pool_size: int = 5000
    selection_strategy: str = "power"  # "top_k", "power", "proportional"
    power_beta: float = 1.0  # AL4PDE default
    
    # Top-k ratio: fraction of samples selected by uncertainty (rest is random)
    # Set to 0.7 to avoid overfitting (70% top-k, 30% random exploration)
    top_k_ratio: float = 0.7
    
    # Rollout: fraction of trajectory (0.5 = 50% of trajectory for faster computation)
    rollout_steps_rel: float = 0.5
    
    batch_pred_size: int = 512  # Increased for A100 GPUs (no gradient)
    use_normalized_uncertainty: bool = True
    uncertainty_aggregation: str = "mean"  # "mean", "max", "sum"
    
    # Hybrid mode: combine breed ratio with SBAL
    hybrid_breed_ratio: float = 0.0  # 0.0 = pure SBAL, 1.0 = pure breed


def parse_sbal_config(config_dict: Dict[str, Any]) -> SBALConfig:
    """Parse SBAL configuration from config dictionary."""
    sbal_cfg = config_dict.get("sbal_config", {})
    return SBALConfig(
        pool_size=sbal_cfg.get("pool_size", 10000),
        selection_strategy=sbal_cfg.get("selection_strategy", "power"),
        power_beta=sbal_cfg.get("power_beta", 1.0),
        top_k_ratio=sbal_cfg.get("top_k_ratio", 0.7),
        rollout_steps_rel=sbal_cfg.get("rollout_steps_rel", 1.0),
        batch_pred_size=sbal_cfg.get("batch_pred_size", 512),
        use_normalized_uncertainty=sbal_cfg.get("use_normalized_uncertainty", True),
        uncertainty_aggregation=sbal_cfg.get("uncertainty_aggregation", "mean"),
        hybrid_breed_ratio=sbal_cfg.get("hybrid_breed_ratio", 0.0),
    )


class SBALParameterSelector:
    """Pool-based parameter selector using ensemble uncertainty.
    
    This class implements the core SBAL logic:
    1. Sample a large pool of candidate parameters uniformly
    2. Compute uncertainty for each candidate using ensemble predictions
    3. Select top-k uncertain candidates for simulation
    
    Uses pool_size=10k (reduced from AL4PDE's 100k for efficiency), full trajectory rollout.
    """
    
    def __init__(
        self,
        config: SBALConfig,
        l_bounds: List[float],
        u_bounds: List[float],
        scenario,
        seed: int = 42,
        trajectory_length: int = 30,  # Default trajectory length
    ):
        """Initialize SBAL selector.
        
        Args:
            config: SBAL configuration
            l_bounds: Lower bounds for each parameter dimension
            u_bounds: Upper bounds for each parameter dimension
            scenario: APEBench scenario (for grid generation)
            seed: Random seed for reproducibility
            trajectory_length: Total trajectory length (for relative rollout steps)
        """
        self.config = config
        self.l_bounds = np.array(l_bounds, dtype=np.float32)
        self.u_bounds = np.array(u_bounds, dtype=np.float32)
        self.nb_params = len(l_bounds)
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trajectory_length = trajectory_length
        
        # Compute actual rollout steps
        self._actual_rollout_steps = self._compute_rollout_steps()
        
        # Statistics tracking
        self.selection_count = 0
        self.total_pool_generated = 0
        self.uncertainty_history: List[float] = []
        
    def generate_candidate_pool(self, n_candidates: int) -> np.ndarray:
        """Generate uniformly distributed candidate parameters.
        
        Args:
            n_candidates: Number of candidates to generate
            
        Returns:
            Array of shape (n_candidates, nb_params) with parameters in bounds
        """
        # Uniform sampling in [0, 1]
        normalized = self.rng.random((n_candidates, self.nb_params), dtype=np.float32)
        # Scale to bounds
        candidates = normalized * (self.u_bounds - self.l_bounds) + self.l_bounds
        self.total_pool_generated += n_candidates
        return candidates
    
    def _compute_rollout_steps(self) -> int:
        """Compute number of rollout steps as fraction of trajectory.
        
        AL4PDE default: rollout_steps_rel=1.0 (full trajectory)
        """
        steps = int(self.trajectory_length * self.config.rollout_steps_rel)
        return max(1, steps)
    
    def set_trajectory_length(self, length: int) -> None:
        """Update trajectory length and recompute rollout steps.
        
        Call this after scenario initialization to set correct trajectory length.
        """
        self.trajectory_length = length
        self._actual_rollout_steps = self._compute_rollout_steps()
        logger.info(
            f"[SBAL] Trajectory length={length}, "
            f"rollout_steps={self._actual_rollout_steps} "
            f"(rel={self.config.rollout_steps_rel})"
        )
    
    def generate_ic_from_params(
        self,
        params: np.ndarray,
        ic_generator,
    ) -> torch.Tensor:
        """Generate initial conditions from parameter vectors.
        
        Args:
            params: Parameter array of shape (batch, nb_params)
            ic_generator: IC generator function that takes params and returns ICs
            
        Returns:
            Tensor of initial conditions [batch, channels, *spatial_dims]
        """
        # This should use the scenario's IC generation
        ics = []
        for p in params:
            ic = ic_generator(p)
            ics.append(ic)
        return torch.stack(ics, dim=0)

    @torch.no_grad()
    def compute_uncertainty_batch(
        self,
        model,
        initial_states: torch.Tensor,
        grid: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute uncertainty for a batch of initial conditions.
        
        Uses ensemble disagreement as uncertainty measure.
        
        Args:
            model: Ensemble model with forward_with_uncertainty method
            initial_states: Initial conditions [batch, channels, *spatial]
            grid: Grid tensor [1, *spatial, dims]
            physics: Optional physics conditioning [batch, physics_dim]
            
        Returns:
            Per-sample uncertainty [batch]
        """
        from apebench_online.models.ensemble import EnsembleModel
        
        batch_size = initial_states.shape[0]
        device = initial_states.device
        
        # Expand grid for batch
        if grid.shape[0] == 1:
            grid = grid.expand(batch_size, -1, -1, -1)
        
        # Check if model is an ensemble
        base_model = model.module if hasattr(model, 'module') else model
        if not isinstance(base_model, EnsembleModel):
            logger.warning(
                "SBAL requires an ensemble model for uncertainty estimation. "
                "Falling back to random selection."
            )
            return torch.ones(batch_size, device=device)
        
        # Rollout and accumulate uncertainty
        current_state = initial_states
        total_uncertainty = torch.zeros(batch_size, device=device)
        
        # Use computed rollout steps (supports AL4PDE-style relative rollout)
        num_steps = self._actual_rollout_steps
        
        for step in range(num_steps):
            pred, uncertainty = base_model.forward_with_uncertainty(
                current_state, grid, physics
            )
            
            # Aggregate uncertainty based on config
            if self.config.uncertainty_aggregation == "max":
                total_uncertainty = torch.max(total_uncertainty, uncertainty)
            elif self.config.uncertainty_aggregation == "sum":
                total_uncertainty = total_uncertainty + uncertainty
            else:  # mean (default)
                total_uncertainty = total_uncertainty + uncertainty / num_steps
            
            current_state = pred
        
        return total_uncertainty

    @torch.no_grad()
    def compute_all_uncertainties(
        self,
        model,
        candidates: np.ndarray,
        ic_generator,
        grid_tensor: torch.Tensor,
        physics_extractor=None,
    ) -> np.ndarray:
        """Compute uncertainty for all candidates in batches.
        
        Args:
            model: Ensemble model
            candidates: Parameter array [n_candidates, nb_params]
            ic_generator: Function to generate ICs from params (single sample)
            grid_tensor: Grid tensor for model
            physics_extractor: Optional function to extract physics from params (single sample)
            
        Returns:
            Uncertainty array [n_candidates]
        """
        n_candidates = len(candidates)
        batch_size = self.config.batch_pred_size
        uncertainties = []
        
        logger.info(f"[SBAL] Computing uncertainties for {n_candidates} candidates...")
        t_start = time.time()
        
        for i in range(0, n_candidates, batch_size):
            batch_params = candidates[i:i + batch_size]
            
            # Generate ICs for this batch
            batch_ics = self.generate_ic_from_params(batch_params, ic_generator)
            batch_ics = batch_ics.to(self.device)
            
            # Get physics for batch if needed - extract per-sample
            batch_physics = None
            if physics_extractor is not None:
                physics_list = []
                for p in batch_params:
                    phys = physics_extractor(p)
                    if phys is not None:
                        physics_list.append(phys)
                if physics_list:
                    batch_physics = torch.stack(physics_list, dim=0).to(self.device)
            
            # Compute uncertainty
            batch_unc = self.compute_uncertainty_batch(
                model, batch_ics, grid_tensor.to(self.device), batch_physics
            )
            uncertainties.append(batch_unc.cpu().numpy())
        
        all_uncertainties = np.concatenate(uncertainties)
        
        elapsed = time.time() - t_start
        logger.info(
            f"[SBAL] Uncertainty computation done in {elapsed:.2f}s "
            f"(mean={all_uncertainties.mean():.4e}, std={all_uncertainties.std():.4e})"
        )
        
        return all_uncertainties

    def select_top_k(self, uncertainties: np.ndarray, k: int) -> np.ndarray:
        """Select indices mixing top-k uncertain samples with random exploration.
        
        Uses top_k_ratio to determine the mix:
        - top_k_ratio fraction selected by highest uncertainty
        - (1 - top_k_ratio) fraction selected randomly from remaining pool
        
        This prevents overfitting to high-uncertainty regions.
        """
        n = len(uncertainties)
        ratio = self.config.top_k_ratio
        
        # Number of samples to select by uncertainty vs random
        k_by_uncertainty = int(k * ratio)
        k_random = k - k_by_uncertainty
        
        if k_by_uncertainty >= n or k_random == 0:
            # All by uncertainty (original behavior)
            return np.argsort(uncertainties)[-k:][::-1]
        
        # Select top-k by uncertainty
        sorted_indices = np.argsort(uncertainties)[::-1]  # Descending
        top_k_indices = sorted_indices[:k_by_uncertainty]
        
        # Select random from remaining pool
        remaining_indices = sorted_indices[k_by_uncertainty:]
        if len(remaining_indices) >= k_random:
            random_indices = self.rng.choice(
                remaining_indices, size=k_random, replace=False
            )
        else:
            random_indices = remaining_indices
        
        return np.concatenate([top_k_indices, random_indices])
    
    def select_power_sampling(
        self, uncertainties: np.ndarray, k: int
    ) -> np.ndarray:
        """Select samples with probability proportional to uncertainty^beta."""
        weights = np.power(np.maximum(uncertainties, 1e-10), self.config.power_beta)
        probs = weights / weights.sum()
        return self.rng.choice(len(uncertainties), size=k, replace=False, p=probs)
    
    def select_proportional(self, uncertainties: np.ndarray, k: int) -> np.ndarray:
        """Select samples with probability proportional to uncertainty."""
        probs = uncertainties / (uncertainties.sum() + 1e-10)
        return self.rng.choice(len(uncertainties), size=k, replace=False, p=probs)
    
    def select_with_diversity(
        self,
        candidates: np.ndarray,
        uncertainties: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Select samples balancing uncertainty and diversity (coreset-style).
        
        Uses a greedy algorithm that at each step selects the candidate that
        maximizes: α * normalized_uncertainty + (1-α) * min_distance_to_selected
        """
        n = len(candidates)
        if k >= n:
            return np.arange(n)
        
        alpha = 1.0 - self.config.diversity_weight
        
        # Normalize uncertainties to [0, 1]
        unc_min, unc_max = uncertainties.min(), uncertainties.max()
        if unc_max > unc_min:
            unc_norm = (uncertainties - unc_min) / (unc_max - unc_min)
        else:
            unc_norm = np.ones_like(uncertainties)
        
        # Normalize candidates for distance computation
        cand_norm = (candidates - self.l_bounds) / (self.u_bounds - self.l_bounds + 1e-10)
        
        selected = []
        remaining = set(range(n))
        
        # Start with highest uncertainty sample
        first = int(np.argmax(uncertainties))
        selected.append(first)
        remaining.remove(first)
        
        # Greedily select remaining
        for _ in range(k - 1):
            best_score = -np.inf
            best_idx = -1
            
            for idx in remaining:
                # Min distance to already selected
                distances = np.linalg.norm(
                    cand_norm[selected] - cand_norm[idx], axis=1
                )
                min_dist = distances.min()
                
                # Combined score
                score = alpha * unc_norm[idx] + (1 - alpha) * min_dist
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return np.array(selected)
    
    def select_parameters(
        self,
        model,
        n_to_select: int,
        ic_generator,
        grid_tensor: torch.Tensor,
        physics_extractor=None,
        tb_logger=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Main entry point: select n_to_select parameters using SBAL.
        
        Args:
            model: Ensemble model for uncertainty estimation
            n_to_select: Number of parameters to select for simulation
            ic_generator: Function to generate ICs from parameter vectors
            grid_tensor: Grid tensor for model input
            physics_extractor: Optional function to extract physics from params
            tb_logger: Optional tensorboard logger
            
        Returns:
            Tuple of (selected_parameters, uncertainties)
        """
        self.selection_count += 1
        
        # Generate candidate pool
        # Use configured pool_size, with minimum of 3x samples to select for diversity
        pool_size = max(self.config.pool_size, n_to_select * 3)
        candidates = self.generate_candidate_pool(pool_size)
        
        logger.info(
            f"[SBAL] Selection {self.selection_count}: pool_size={pool_size}, "
            f"selecting {n_to_select} samples"
        )
        
        # Compute uncertainties - pass physics_extractor to be called per-sample
        uncertainties = self.compute_all_uncertainties(
            model, candidates, ic_generator, grid_tensor, physics_extractor
        )
        
        # Select based on strategy
        if self.config.selection_strategy == "top_k":
            selected_idx = self.select_top_k(uncertainties, n_to_select)
        elif self.config.selection_strategy == "power":
            selected_idx = self.select_power_sampling(uncertainties, n_to_select)
        elif self.config.selection_strategy == "proportional":
            selected_idx = self.select_proportional(uncertainties, n_to_select)
        else:
            logger.warning(f"Unknown selection strategy: {self.config.selection_strategy}")
            selected_idx = self.rng.choice(pool_size, size=n_to_select, replace=False)
        
        selected_params = candidates[selected_idx]
        selected_uncertainties = uncertainties[selected_idx]
        
        # Log statistics
        self.uncertainty_history.append(float(selected_uncertainties.mean()))
        
        if tb_logger is not None:
            tb_logger.log_scalar(
                "SBAL/selected_mean_uncertainty",
                selected_uncertainties.mean(),
                self.selection_count,
            )
            tb_logger.log_scalar(
                "SBAL/pool_mean_uncertainty",
                uncertainties.mean(),
                self.selection_count,
            )
            tb_logger.log_scalar(
                "SBAL/uncertainty_ratio",
                selected_uncertainties.mean() / (uncertainties.mean() + 1e-10),
                self.selection_count,
            )
        
        logger.info(
            f"[SBAL] Selected {n_to_select} samples with mean uncertainty "
            f"{selected_uncertainties.mean():.4e} (pool mean: {uncertainties.mean():.4e})"
        )
        
        return selected_params, selected_uncertainties


class SBALBreederAdapter:
    """Adapter to use SBAL within the OGASBreeder framework.
    
    This allows SBAL to be used as a drop-in replacement for the generative
    model-based breeding in OGASBreeder, while keeping the same interface.
    """
    
    def __init__(
        self,
        sbal_selector: SBALParameterSelector,
        model,
        ic_generator,
        grid_tensor: torch.Tensor,
        physics_extractor=None,
    ):
        self.sbal_selector = sbal_selector
        self.model = model
        self.ic_generator = ic_generator
        self.grid_tensor = grid_tensor
        self.physics_extractor = physics_extractor
        self.metric_logger = None
    
    def set_tb_logger(self, tb_logger):
        self.metric_logger = tb_logger
    
    def generate_bred_parameters(self, n_to_generate: int) -> np.ndarray:
        """Generate parameters using SBAL (for compatibility with OGASBreeder).
        
        Args:
            n_to_generate: Number of parameters to generate
            
        Returns:
            Array of selected parameters [n_to_generate, nb_params]
        """
        selected_params, _ = self.sbal_selector.select_parameters(
            self.model,
            n_to_generate,
            self.ic_generator,
            self.grid_tensor,
            self.physics_extractor,
            self.metric_logger,
        )
        return selected_params
