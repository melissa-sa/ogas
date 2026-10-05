import logging
from typing import Dict, Any, List, Optional, Callable
from typing_extensions import override

import numpy as np

from melissa.server.deep_learning.active_sampling.breeder import (
    ExperimentBreeder,
    BreedMetadata,
)
from melissa.server.parameters import ParameterSamplerType

from apebench_online.sampler.ic_sampler import (
    BaseCustomSamplerMixIn,
    _extract_base_sampler_kwargs,
)
from apebench_online.sbal.sbal_sampler import (
    SBALParameterSelector,
    SBALConfig,
)

import numpy as np
logger = logging.getLogger(__name__)


class ICSamplerSBAL(BaseCustomSamplerMixIn, ExperimentBreeder):
    """IC Sampler using SBAL (Simulation-Based Active Learning).
    
    This sampler uses ensemble uncertainty to select which simulation
    parameters to run next, implementing pool-based active learning.
    
    Unlike OGASBreeder which trains a generative model to propose parameters,
    SBAL evaluates uncertainty on a large pool of uniformly sampled candidates
    and selects those with highest uncertainty.
    
    Attributes:
        sbal_selector: The SBAL parameter selector (set by server)
        model_ref: Reference to the ensemble model (set by server)
        grid_tensor_ref: Reference to grid tensor (set by server)
        ic_generator_ref: Reference to IC generator function (set by server)
    """
    
    def __init__(
        self,
        ic_config: Dict[str, Any],
        is_valid: bool = False,
        sbal_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize SBAL sampler.
        
        Args:
            ic_config: IC configuration dictionary
            is_valid: Whether this is for validation (disables SBAL)
            sbal_config: SBAL-specific configuration
            **kwargs: Additional arguments passed to ExperimentBreeder
        """
        base_kwargs = _extract_base_sampler_kwargs(kwargs)
        
        # Initialize base classes
        ExperimentBreeder.__init__(
            self,
            nb_params=kwargs.get("nb_params", 1),
            nb_sims=kwargs.get("nb_sims", 1000),
            l_bounds=kwargs.get("l_bounds", []),
            u_bounds=kwargs.get("u_bounds", []),
            seed=kwargs.get("seed", 42),
            dtype=kwargs.get("dtype", np.float32),
            non_breed_sampler_t=kwargs.get(
                "non_breed_sampler_t", 
                ParameterSamplerType.RANDOM_UNIFORM
            ),
        )
        BaseCustomSamplerMixIn.__init__(
            self, 
            ic_config=ic_config, 
            is_valid=is_valid, 
            **base_kwargs
        )
        
        # SBAL-specific config
        self.sbal_config = sbal_config or {}
        self.use_sbal = not is_valid  # Disable for validation
        
        # These will be set by the server after model initialization
        self.sbal_selector = None
        self.model_ref: Optional[Callable] = None
        self.grid_tensor_ref = None
        self.ic_generator_ref: Optional[Callable] = None
        self.physics_extractor_ref: Optional[Callable] = None
        self.metric_logger_ref = None
        
        # Breeding state
        self.current_generation = 0
        self.temp_children_metadata_list: List[BreedMetadata] = []
        self.parameters_is_bred: np.ndarray = np.full(
            self.nb_sims, False, dtype=np.bool_
        )
        self.last_submitted_sim_id: int = 0
        
        # Breeding ratio schedule (for hybrid mode)
        self._breed_ratios = self._compute_breed_ratios()
        self._current_ratio_idx = 0
        
        logger.info(
            f"[SBAL Sampler] Initialized with {self.nb_sims} sims, "
            f"pool_size={self.sbal_config.get('pool_size', 10000)}"
        )
    
    def _compute_breed_ratios(self) -> np.ndarray:
        """Compute breeding ratio schedule."""
        start_ratio = self.sbal_config.get("start_breed_ratio", 0.0)
        end_ratio = self.sbal_config.get("end_breed_ratio", 0.7)
        breakpoint = self.sbal_config.get("breakpoint", 4)
        return np.linspace(start_ratio, end_ratio, max(breakpoint, 2))
    
    def set_model_reference(
        self,
        model,
        grid_tensor,
        ic_generator: Callable,
        physics_extractor: Optional[Callable] = None,
        tb_logger=None,
    ) -> None:
        """Set references to model and generators for SBAL.
        
        This should be called by the server after model initialization.
        
        Args:
            model: The ensemble model for uncertainty estimation
            grid_tensor: Grid tensor for model input
            ic_generator: Function to generate ICs from parameter vectors
            physics_extractor: Optional function to extract physics conditioning
            tb_logger: Tensorboard logger for metrics
        """
        self.model_ref = model
        self.grid_tensor_ref = grid_tensor
        self.ic_generator_ref = ic_generator
        self.physics_extractor_ref = physics_extractor
        self.metric_logger_ref = tb_logger
        
        # Initialize SBAL selector if not done
        if self.sbal_selector is None and self.use_sbal:

            config = SBALConfig(
                pool_size=self.sbal_config.get("pool_size", 100000),
                selection_strategy=self.sbal_config.get("selection_strategy", "power"),
                power_beta=self.sbal_config.get("power_beta", 1.0),
                top_k_ratio=self.sbal_config.get("top_k_ratio", 0.7),
                rollout_steps_rel=self.sbal_config.get("rollout_steps_rel", 1.0),
                batch_pred_size=self.sbal_config.get("batch_pred_size", 128),
                use_normalized_uncertainty=self.sbal_config.get(
                    "use_normalized_uncertainty", True
                ),
                uncertainty_aggregation=self.sbal_config.get(
                    "uncertainty_aggregation", "mean"
                ),
                hybrid_breed_ratio=self.sbal_config.get("hybrid_breed_ratio", 0.0),
            )
            self.sbal_selector = SBALParameterSelector(
                config=config,
                l_bounds=self.l_bounds,
                u_bounds=self.u_bounds,
                scenario=None,  # Not needed for param generation
                seed=self.seed,
            )
            logger.info("[SBAL Sampler] SBAL selector initialized")
    
    @property
    def current_breed_ratio(self) -> float:
        """Get current breeding ratio."""
        idx = min(self._current_ratio_idx, len(self._breed_ratios) - 1)
        return float(self._breed_ratios[idx])
    
    @override
    def next_parameters(self, max_breeding_count: int = -1) -> None:
        """Generate next batch of parameters using SBAL.
        
        This overrides ExperimentBreeder.next_parameters to use
        SBAL uncertainty-based selection instead of mutation/crossover.
        
        Args:
            max_breeding_count: Maximum number of parameters to generate
        """
        self.current_generation += 1
        logger.info(
            f"[SBAL] Generation {self.current_generation}: "
            f"generating next parameters (breed_ratio={self.current_breed_ratio:.2f})"
        )
        
        # Compute number of children to generate
        remaining = self.nb_sims - (self.last_submitted_sim_id + 1)
        if max_breeding_count <= 0:
            nb_children = remaining
        else:
            nb_children = min(max_breeding_count, remaining)
        
        if nb_children <= 0:
            logger.warning("[SBAL] No simulations remaining to generate")
            self.temp_children_metadata_list = []
            return
        
        # Split between SBAL (bred) and random (non-bred)
        nb_bred = int(nb_children * self.current_breed_ratio)
        nb_random = nb_children - nb_bred
        
        bred_params = self._generate_sbal_parameters(nb_bred)
        random_params = self.get_non_breed_samples(nb_samples=nb_random)
        
        logger.info(
            f"[SBAL] Generated {len(bred_params)} SBAL + {len(random_params)} random"
        )
        
        # Combine and shuffle
        all_params = np.vstack([bred_params, random_params]) if nb_bred > 0 else random_params
        is_bred_mask = np.array([True] * nb_bred + [False] * nb_random)
        
        shuffle_idx = np.random.permutation(nb_children)
        self._stage_parameters(all_params[shuffle_idx], is_bred_mask[shuffle_idx])
    
    def _generate_sbal_parameters(self, n_to_generate: int) -> np.ndarray:
        """Generate parameters using SBAL uncertainty selection.
        
        Args:
            n_to_generate: Number of parameters to generate
            
        Returns:
            Array of selected parameters [n_to_generate, nb_params]
        """
        if n_to_generate == 0:
            return np.array([]).reshape(0, self.nb_params)
        
        # Check if SBAL is properly initialized
        if self.sbal_selector is None or self.model_ref is None:
            logger.warning(
                "[SBAL] Selector or model not initialized, falling back to random"
            )
            return self.get_non_breed_samples(nb_samples=n_to_generate)
        
        if self.ic_generator_ref is None:
            logger.warning(
                "[SBAL] IC generator not set, falling back to random"
            )
            return self.get_non_breed_samples(nb_samples=n_to_generate)
        
        # Use SBAL selector
        try:
            params, uncertainties = self.sbal_selector.select_parameters(
                model=self.model_ref,
                n_to_select=n_to_generate,
                ic_generator=self.ic_generator_ref,
                grid_tensor=self.grid_tensor_ref,
                physics_extractor=self.physics_extractor_ref,
                tb_logger=self.metric_logger_ref,
            )
            return params
        except Exception as e:
            logger.error(f"[SBAL] Selection failed: {e}, falling back to random")
            return self.get_non_breed_samples(nb_samples=n_to_generate)
    
    def _stage_parameters(
        self, 
        params: np.ndarray, 
        is_bred_mask: np.ndarray
    ) -> None:
        """Stage parameters in temporary list before concretization.
        
        Args:
            params: Parameter array [n_samples, nb_params]
            is_bred_mask: Boolean mask indicating which are SBAL-selected
        """
        self.temp_children_metadata_list = []
        for i, param_vec in enumerate(params):
            meta = BreedMetadata(
                sim_id=-1,  # Assigned during concretization
                generation=self.current_generation,
                is_bred=bool(is_bred_mask[i]),
                parameters=param_vec.copy(),
            )
            self.temp_children_metadata_list.append(meta)
        
        logger.info(
            f"[SBAL] Staged {len(self.temp_children_metadata_list)} parameters "
            f"for generation {self.current_generation}"
        )
    
    @override
    def concretize_resampled_parameters(
        self, 
        last_submitted_sim_id: int
    ) -> tuple[int, int]:
        """Concretize staged parameters into the main parameter array.
        
        For SBAL, this runs on the training rank (rank 0), not the sampling rank.
        We need to handle the memmap being read-only by reopening it with write access.
        
        Args:
            last_submitted_sim_id: Last submitted simulation ID
            
        Returns:
            Tuple of (first_id, last_id) of concretized parameters
        """
        nb_produced = len(self.temp_children_metadata_list)
        self.last_submitted_sim_id = last_submitted_sim_id
        first_id = last_submitted_sim_id + 1
        last_id = min(self.nb_sims, first_id + nb_produced - 1)
        
        # For SBAL, we may not be on the sampling rank, so we need to 
        # directly write to the memmap file with write access        
        # Open memmap with write access for this operation
        params_memmap = np.memmap(
            filename=self.memmap_file,
            dtype=self.dtype,
            mode="r+",
            shape=(self.nb_sims, self.nb_params),
        )
        
        try:
            for i, child in zip(
                range(first_id, first_id + nb_produced), 
                self.temp_children_metadata_list
            ):
                if i >= self.nb_sims:
                    break
                child.sim_id = i
                self.current_metadata_list[i] = child
                params_memmap[i] = child.parameters
                self.parameters_is_bred[i] = child.is_bred
            
            # Flush the memmap
            params_memmap.flush()
        finally:
            # Clean up
            del params_memmap
        
        # Advance ratio schedule
        self._current_ratio_idx += 1
        
        self.temp_children_metadata_list = []
        # Don't call flush_to_disk() as it's decorated with @sampling_rank_only
        # and we already flushed above
        
        final_last_id = first_id + nb_produced - 1
        return first_id, min(self.nb_sims - 1, final_last_id)
    
    def collect_data(self) -> None:
        """Collect data for SBAL (no-op, SBAL doesn't need training)."""
        pass
    
    def train_sampler(self) -> None:
        """Train sampler (no-op, SBAL uses ensemble uncertainty directly)."""
        pass


def get_sbal_sampler_class():
    """Get the SBAL sampler class for use in config.
    
    Returns:
        ICSamplerSBAL class
    """
    return ICSamplerSBAL
