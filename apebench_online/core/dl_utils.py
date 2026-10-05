import logging
import os
import os.path as osp
import re
import socket
from collections import OrderedDict
import numpy as np

import jax
import jax.numpy as jnp
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as jshard
import equinox as eqx

from melissa.utility.rank_helper import (
    initialize_sampling_rank,
    get_sampling_rank,
    ClusterEnvironment,
)


logger = logging.getLogger("melissa")

jax.config.update("jax_default_matmul_precision", "float32")


def replicate_model(model):
    num_devices = len(jax.devices())
    if num_devices == 1:
        return model, None

    devices = mesh_utils.create_device_mesh((num_devices, 1))
    sharding = jshard.PositionalSharding(devices)
    replicated = sharding.replicate()
    model = eqx.filter_shard(model, replicated)

    return model, sharding


def loss_fn(model, x, y):
    y_pred = jax.vmap(model)(x)
    reduce_axes = tuple(range(1, y_pred.ndim))
    loss_per_sample = jnp.mean(jnp.square(y_pred - y), axis=reduce_axes)
    batch_loss = jnp.mean(loss_per_sample)
    return batch_loss, loss_per_sample


def loss_fn_two_step(model, x, y):
    y_pred_1 = jax.vmap(model)(x)
    y_pred_2 = jax.vmap(model)(y_pred_1)
    reduce_axes = tuple(range(1, y_pred_2.ndim))
    loss_per_sample = jnp.mean(jnp.square(y_pred_2 - y), axis=reduce_axes)
    batch_loss = jnp.mean(loss_per_sample)
    return batch_loss, loss_per_sample


@eqx.filter_jit(donate="all")
def update_fn(model, optimizer, x, y, opt_state, sharding=None):
    if sharding is not None:
        replicated = sharding.replicate()
        model, opt_state = eqx.filter_shard((model, opt_state), replicated)

    (loss, loss_per_sample), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, x, y)

    updates, new_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    if sharding is not None:
        new_model, new_state = eqx.filter_shard((new_model, new_state), replicated)

    return (new_model, new_state, loss, loss_per_sample)


@eqx.filter_jit(donate="all")
def update_fn_two_step(model, optimizer, x, y, opt_state, sharding=None):
    if sharding is not None:
        replicated = sharding.replicate()
        model, opt_state = eqx.filter_shard((model, opt_state), replicated)

    (loss, loss_per_sample), grads = eqx.filter_value_and_grad(loss_fn_two_step, has_aux=True)(
        model, x, y
    )

    updates, new_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    if sharding is not None:
        new_model, new_state = eqx.filter_shard((new_model, new_state), replicated)

    return (new_model, new_state, loss, loss_per_sample)


class AutoregressiveTrajectoryDataset:
    """Loads trajectories stored in numpy files, specific for validation.

    Method `get_pairs` :
        Returns necessary time steps data for the given batch range.
    Method `get_rollout` :
        Returns the rollout timesteps for the given batch range of trajectory samples.
    """

    _CACHE_SIZE = 40

    def __init__(
        self,
        data_path: str,
        nb_time_steps: int,
        num_samples: int | None = None,
        rank: int = 0,
        comm_size: int = 1,
    ) -> None:
        self.nb_time_steps = nb_time_steps
        self.rank = rank
        self.comm_size = comm_size
        self._data_path = data_path
        self._is_directory = osp.isdir(data_path)
        self._trajectory_major_file = False

        logger.info(f"Loading validation dataset from {data_path}...")

        self.global_num_samples = 0
        self.num_samples = 0
        self.sample_shape: tuple[int, ...] | None = None
        self.validation_dataset = None
        self.trajectory_files: list[str] = []
        self._trajectory_cache: OrderedDict[int, np.memmap] = OrderedDict()
        self._cache_size = self._CACHE_SIZE

        if self._is_directory:
            self._init_from_directory(num_samples)
        else:
            self._init_from_file(num_samples)

        self.dataset_length = self.num_samples * self.nb_time_steps
        self.num_pairs = max(self.num_samples * (self.nb_time_steps - 1), 0)

        if self.sample_shape is None and self.dataset_length == 0:
            self.sample_shape = tuple()

        shape_repr = self.sample_shape if self.sample_shape is not None else "unknown"
        logger.info(
            f"Rank {self.rank}>> Validation data source loaded {self.num_samples} samples, "
            f"{self.num_pairs} pairs, sample shape {shape_repr}."
        )

    def _init_from_file(self, num_samples: int | None) -> None:
        validation_dataset_mmap = np.load(self._data_path, mmap_mode="r")
        self._trajectory_major_file = (
            validation_dataset_mmap.ndim >= 2
            and validation_dataset_mmap.shape[1] == self.nb_time_steps
        )
        if self._trajectory_major_file:
            total_samples = validation_dataset_mmap.shape[0]
        else:
            total_samples = validation_dataset_mmap.shape[0] // self.nb_time_steps
        if num_samples is not None:
            total_samples = min(num_samples, total_samples)

        self.global_num_samples = total_samples
        start_sample, end_sample = self._rank_slice(self.global_num_samples)
        self.num_samples = max(end_sample - start_sample, 0)

        if self._trajectory_major_file:
            start_idx = start_sample
            end_idx = end_sample
        else:
            start_idx = start_sample * self.nb_time_steps
            end_idx = end_sample * self.nb_time_steps

        self.validation_dataset = validation_dataset_mmap[start_idx:end_idx]
        del validation_dataset_mmap

        if self.num_samples > 0:
            if self._trajectory_major_file:
                self.sample_shape = tuple(self.validation_dataset.shape[2:])
            else:
                self.sample_shape = tuple(self.validation_dataset.shape[1:])

    def _init_from_directory(self, num_samples: int | None) -> None:
        candidate_files = self._discover_trajectory_files(self._data_path)
        if not candidate_files:
            raise FileNotFoundError(
                f"No trajectory .npy files found in validation directory {self._data_path}."
            )

        total_available = len(candidate_files)
        if num_samples is not None:
            total_available = min(num_samples, total_available)
            candidate_files = candidate_files[:total_available]

        self.global_num_samples = total_available
        start_sample, end_sample = self._rank_slice(self.global_num_samples)
        self.trajectory_files = candidate_files[start_sample:end_sample]
        self.num_samples = len(self.trajectory_files)

        if self.num_samples > 0:
            first_traj = np.load(self.trajectory_files[0], mmap_mode="r")
            if first_traj.shape[0] != self.nb_time_steps:
                raise ValueError(
                    f"Trajectory {self.trajectory_files[0]} has {first_traj.shape[0]} time steps, "
                    f"expected {self.nb_time_steps}."
                )
            self.sample_shape = tuple(first_traj.shape[1:])

            self._cache_store(0, first_traj)

    def _discover_trajectory_files(self, directory: str) -> list[str]:
        entries = []
        for name in os.listdir(directory):
            full_path = osp.join(directory, name)
            if osp.isfile(full_path) and name.endswith(".npy"):
                # Skip highres files to avoid mixing resolutions
                if "_highres" in name.lower():
                    continue
                entries.append(full_path)

        if not entries:
            return []

        prioritized = [
            path
            for path in entries
            if any(token in osp.basename(path).lower() for token in ("sim", "traj"))
        ]
        if prioritized:
            entries = prioritized

        entries.sort(key=self._trajectory_sort_key)
        return entries

    @staticmethod
    def _trajectory_sort_key(path: str):
        name = osp.splitext(osp.basename(path))[0]
        match = re.search(r"(\d+)", name)
        if match:
            return int(match.group(1))
        return name

    def _rank_slice(self, total_elements: int) -> tuple[int, int]:
        if total_elements == 0 or self.comm_size <= 1:
            return 0, total_elements

        base = total_elements // self.comm_size
        remainder = total_elements % self.comm_size

        start = self.rank * base + min(self.rank, remainder)
        end = start + base + (1 if self.rank < remainder else 0)
        return start, end

    def _cache_store(self, sample_idx: int, trajectory: np.ndarray) -> None:
        self._trajectory_cache[sample_idx] = trajectory
        self._trajectory_cache.move_to_end(sample_idx)
        while len(self._trajectory_cache) > self._cache_size:
            _, old = self._trajectory_cache.popitem(last=False)
            del old

    def _get_trajectory(self, sample_idx: int) -> np.ndarray:
        if sample_idx < 0 or sample_idx >= self.num_samples:
            raise IndexError(
                f"Sample index {sample_idx} out of range for {self.num_samples} samples."
            )

        cached = self._trajectory_cache.get(sample_idx)
        if cached is not None:
            self._trajectory_cache.move_to_end(sample_idx)
            return cached

        path = self.trajectory_files[sample_idx]
        trajectory = np.load(path, mmap_mode="r")
        if trajectory.shape[0] != self.nb_time_steps:
            raise ValueError(
                f"Trajectory {path} has {trajectory.shape[0]} time steps, expected {self.nb_time_steps}."
            )
        self.sample_shape = (
            tuple(trajectory.shape[1:]) if self.sample_shape is None else self.sample_shape
        )
        self._cache_store(sample_idx, trajectory)
        return trajectory


    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if not self._is_directory:
            if not self._trajectory_major_file:
                return np.array(self.validation_dataset[idx], dtype=np.float32, copy=False)

            indices, is_scalar = self._normalize_indices(idx)
            if not indices:
                return self._empty_like(0) if not is_scalar else self._empty_like_scalar()

            batch = []
            for index in indices:
                if index < 0 or index >= self.dataset_length:
                    raise IndexError(
                        f"Index {index} out of range for dataset of length {self.dataset_length}."
                    )
                sample_idx = index // self.nb_time_steps
                timestep_idx = index % self.nb_time_steps
                batch.append(
                    np.array(
                        self.validation_dataset[sample_idx, timestep_idx],
                        dtype=np.float32,
                        copy=False,
                    )
                )

            stacked = np.stack(batch, axis=0)
            if is_scalar:
                return stacked[0]
            return stacked

        indices, is_scalar = self._normalize_indices(idx)
        if not indices:
            return self._empty_like(0) if not is_scalar else self._empty_like_scalar()

        batch = []
        for index in indices:
            if index < 0 or index >= self.dataset_length:
                raise IndexError(
                    f"Index {index} out of range for dataset of length {self.dataset_length}."
                )
            sample_idx = index // self.nb_time_steps
            timestep_idx = index % self.nb_time_steps
            trajectory = self._get_trajectory(sample_idx)
            batch.append(np.array(trajectory[timestep_idx], dtype=np.float32, copy=False))

        stacked = np.stack(batch, axis=0)

        if is_scalar:
            return stacked[0]
        return stacked

    def _normalize_indices(self, idx) -> tuple[list[int], bool]:
        if isinstance(idx, slice):
            rng = range(*idx.indices(self.dataset_length))
            return list(rng), False

        arr = np.asarray(idx)
        if arr.ndim == 0:
            return [int(arr)], True

        if arr.size == 0:
            return [], False

        if not np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int64)

        return arr.tolist(), False

    def _empty_like(self, length: int) -> np.ndarray:
        shape = (length,)
        if self.sample_shape:
            shape = (length, *self.sample_shape)
        return np.empty(shape, dtype=np.float32)

    def _empty_like_scalar(self) -> np.ndarray:
        if self.sample_shape:
            return np.empty(self.sample_shape, dtype=np.float32)
        return np.empty((), dtype=np.float32)

    def get_pairs(self, batch_l: int, batch_r: int) -> tuple[list[int], list[int], list[int]]:
        """
        Returns necessary time steps data for the given batch range.
        This function is optimized, as it does not load timesteps twice for input and ouput.
        For large datasets, instead of loading a whole trajectory for each simulation sample, it treats
        time steps pairs as an item of a batch and loads only necessary time steps for the given batch range.
        Use as follows:
        ```
        u_prev_indices, u_next_indices = dataset.get_pairs(i, i + BATCH_SIZE)
        u_prev = dataset[u_prev_indices].to(device)
        u_pred = model(u_prev).detach().cpu()
        valid_loss = error(u_pred, dataset[u_next_indices])
        ```
        Returns:
            - indices of the previous time step
            - indices of the next time step
            - simulation indices
        """

        def convert_to_array_index(batch_id):
            return min(
                batch_id + (batch_id // (self.nb_time_steps - 1)),
                self.num_pairs + self.num_samples,
            )

        u_prev_indices: list[int] = []
        u_next_indices: list[int] = []
        for i in range(convert_to_array_index(batch_l), convert_to_array_index(batch_r)):
            if (i + 1) % self.nb_time_steps != 0:
                # add only if not the last time step
                u_prev_indices.append(i)
                u_next_indices.append(i + 1)

        return u_prev_indices, u_next_indices

    def get_rollout(
        self, batch_l: int, batch_r: int, rollout_size: int | None = None
    ) -> tuple[list[int], list[list[int]]]:
        """
        Returns the rollout timesteps for the given batch range of trajectory samples.
        Use as follows:
        ```
        ic_idx, rollout_idx = dataset.get_rollout(i, i + BATCH_SIZE, rollout_size=ROLL_SIZE)
        u_step = dataset[ic_idx].to(device)
        rollout_losses = []
        for roll_idx in rollout_idx:
            u_step = model(u_step)
            rollout_losses.append(error(u_step.detach().cpu(), dataset[roll_idx]))
        ```
        This way avoids unnecessary memory usage by predicting the rollout step by step instead of the whole rollout at once.
        """

        if rollout_size is None:
            rollout_size = self.nb_time_steps
        assert (
            0 < rollout_size < self.nb_time_steps
        ), f"Rollout size must be at least 1 and not more than {self.nb_time_steps - 1}, but got {rollout_size}."
        assert batch_l >= 0, f"Rollout index must be non negative {batch_l} < 0."
        assert (
            batch_r <= self.num_samples
        ), f"Rollout index must be not more than the number of samples, but got {batch_r} > {self.num_samples}."

        ic_idx = [i * self.nb_time_steps for i in range(batch_l, batch_r)]
        rollout_idx = []
        for r in range(1, rollout_size + 1):
            rollout_idx.append([i + r for i in ic_idx])
        return ic_idx, rollout_idx

    def _prepare_batch_generator(self, batch_size: int, rollout_size: int):
        self.batched_indices = []
        for i in range(0, self.num_pairs, batch_size):
            j = min(self.num_pairs, i + batch_size)
            self.batched_indices.append(self.get_pairs(i, j))
        if rollout_size != -1:
            self.batched_indices_rollout = []
            for i in range(0, self.num_samples, batch_size):
                j = min(self.num_samples, i + batch_size)
                self.batched_indices_rollout.append(self.get_rollout(i, j, rollout_size))

    def batch_generator(self, rollout_size: int = -1):
        if rollout_size != -1:  # want rollout batches
            return self.batched_indices_rollout
        else:  # want 1to1 batches
            return self.batched_indices


def load_validation_dataset(
    validation_dir,
    validation_file,
    nb_time_steps,
    batch_size,
    rollout_size=-1,
    num_samples=None,
    rank=0,
    comm_size=1,
):
    if validation_dir is None:
        return None

    valid_dataset = None
    valid_parameters = None
    valid_dataloader = None
    valid_dataloader_rollout = None
    validation_dir = osp.expandvars(validation_dir)
    # Find trajectory and parameter files
    files = os.listdir(validation_dir)

    traj_source = None
    if validation_file is not None:
        candidate = osp.expandvars(validation_file)
        if not osp.isabs(candidate):
            candidate = osp.join(validation_dir, candidate)
        if osp.isdir(candidate):
            traj_source = candidate
            logger.info(f"Found trajectory directory: {traj_source}")
        elif osp.isfile(candidate):
            traj_source = candidate
            logger.info(f"Found trajectory file: {traj_source}")
        else:
            logger.warning(f"Validation file {validation_file} not found in {validation_dir}.")

    params_path = None
    for file in files:
        if "param" in file.lower() and file.endswith(".npy") and params_path is None:
            params_path = osp.join(validation_dir, file)
            logger.info(f"Found parameter file: {params_path}")

    if traj_source is None:
        candidate_dirs = []
        for name in files:
            full_path = osp.join(validation_dir, name)
            if osp.isdir(full_path):
                try:
                    has_npy = any(entry.endswith(".npy") for entry in os.listdir(full_path))
                except FileNotFoundError:
                    has_npy = False
                if has_npy:
                    candidate_dirs.append(full_path)
        if candidate_dirs:
            candidate_dirs.sort(
                key=lambda p: (
                    not osp.basename(p).lower().startswith("traj"),
                    osp.basename(p),
                )
            )
            traj_source = candidate_dirs[0]
            logger.info(f"Found trajectory directory: {traj_source}")

    if traj_source is None:
        trajectory_files = [
            file for file in files if file.endswith(".npy") and "param" not in file.lower()
        ]
        if len(trajectory_files) > 1:
            traj_source = validation_dir
            logger.info(
                "Detected multiple trajectory files; using directory "
                f"{traj_source} as validation source."
            )
        elif len(trajectory_files) == 1:
            traj_source = osp.join(validation_dir, trajectory_files[0])
            logger.info(f"Found trajectory file: {traj_source}")

    if params_path is None and traj_source is not None and osp.isdir(traj_source):
        try:
            for file in os.listdir(traj_source):
                if "param" in file.lower() and file.endswith(".npy"):
                    params_path = osp.join(traj_source, file)
                    logger.info(f"Found parameter file inside trajectory directory: {params_path}")
                    break
        except FileNotFoundError:
            pass

    if traj_source is not None and osp.exists(traj_source):
        valid_dataset = AutoregressiveTrajectoryDataset(
            data_path=traj_source,
            nb_time_steps=nb_time_steps,
            num_samples=num_samples,
            rank=rank,
            comm_size=comm_size,
        )
        logger.info(f"Preparing validation dataset with batch size {batch_size}.")
        valid_dataset._prepare_batch_generator(batch_size=batch_size, rollout_size=rollout_size)

        valid_dataloader = valid_dataset.batch_generator()
        if rollout_size != -1:
            valid_dataloader_rollout = valid_dataset.batch_generator(rollout_size=rollout_size)

        # only required on rank 0 for plotting
        if params_path and osp.exists(params_path) and rank == 0:
            valid_parameters = np.load(params_path)[: valid_dataset.global_num_samples]

        logger.info(f"Validation dataset is prepared: {len(valid_dataloader)} batches.")
    else:
        source_display = traj_source if traj_source is not None else validation_dir
        logger.warning(
            f"Validation set not found at {source_display}"
            "\nPlease set validation_directory in configuration."
        )

    return dict(
        dataset=valid_dataset,
        input_parameters=valid_parameters,
        dataloader=valid_dataloader,
        dataloader_rollout=valid_dataloader_rollout,
    )
