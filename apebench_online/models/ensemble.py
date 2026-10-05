from typing import Callable, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch import nn


class AdaptiveGradClip:
    def __init__(self, multiplier: float = 5.0, alpha: float = 0.05, warmup_steps: int = 5, max_norm=None, adaptive=False):
        self.multiplier, self.alpha, self.warmup_steps = multiplier, alpha, warmup_steps
        self.max_norm, self.adaptive = max_norm, adaptive
        self.ema: Optional[float] = None

    def __call__(self, params: Sequence[torch.Tensor], step: int) -> None:
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or not (self.adaptive or self.max_norm is not None):
            return
        norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])))
        if self.adaptive:
            if self.ema is None or step == 0:
                self.ema = norm
            elif step < self.warmup_steps:
                self.ema = max(norm, self.ema)
            else:
                self.ema = (1 - self.alpha) * self.ema + self.alpha * min(norm, self.multiplier * self.ema)
            threshold = self.multiplier * self.ema
        else:
            threshold = self.max_norm
        if norm > threshold:
            torch._foreach_mul_(grads, threshold / (norm + 1e-6))


def disagreement(preds: torch.Tensor) -> torch.Tensor:
    return (preds - preds.mean(0)).pow(2).flatten(2).mean(2).mean(0)


def ensemble_crps(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = preds.shape[0]
    spread = sum((preds[i] - preds[j]).abs() for i in range(m) for j in range(i + 1, m))
    score = (preds - target).abs().mean(0) - spread / m ** 2
    return score.flatten(1).mean(1)


class EnsembleModel(nn.Module):
    def __init__(self, members: List[nn.Module], member_ids: Optional[List[int]] = None, n_members: Optional[int] = None,
                 group=None, prediction_mode: str = "mean", acquisition_signal: str = "uncertainty"):
        super().__init__()
        if prediction_mode not in ("mean", "member0"):
            raise ValueError("prediction_mode must be 'mean' or 'member0'.")
        self.models = nn.ModuleList(members)
        self.member_ids = list(member_ids) if member_ids is not None else list(range(len(members)))
        self.n_models = n_members or len(members)
        self.group = group
        self.world = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0
        if self.world * len(self.models) != self.n_models:
            raise ValueError(f"{self.n_models} members cannot be split evenly over {self.world} ranks.")
        self.prediction_mode = prediction_mode
        self.acquisition_signal = acquisition_signal.lower()
        self.optimizers: List[torch.optim.Optimizer] = []
        self.schedulers: List = []
        self.clippers: List[AdaptiveGradClip] = []

    def init_optimizers(self, optimizer_class, optimizer_kwargs, scheduler_builder: Optional[Callable] = None, **clip_kwargs):
        self.optimizers = [optimizer_class(m.parameters(), **optimizer_kwargs) for m in self.models]
        self.schedulers = [scheduler_builder(o) for o in self.optimizers] if scheduler_builder else []
        self.clippers = [AdaptiveGradClip(**clip_kwargs) for _ in self.models]
        return self.optimizers

    def get_learning_rate(self) -> float:
        return self.optimizers[0].param_groups[0]["lr"] if self.optimizers else 0.0

    def _gather(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None or self.world == 1:
            return t
        parts = [torch.empty_like(t) for _ in range(self.world)]
        dist.all_gather(parts, t.contiguous(), group=self.group)
        return torch.cat(parts)

    def _all_predictions(self, state, grid, physics) -> torch.Tensor:
        return self._gather(torch.stack([m(state, grid, physics) for m in self.models]))

    def _select(self, preds: torch.Tensor) -> torch.Tensor:
        return preds[0] if self.prediction_mode == "member0" else preds.mean(0)

    def training_step(self, state, grid, target, criterion, physics=None, step: int = 0, target2=None, batch_size=None):
        """One optimisation step of every member on the gathered batch.

        `state`, `target`, `physics`, `target2` are this rank's local batch; `grid` is [1, ...]. The global
        batch is the concatenation of the local batches, truncated to `batch_size`. With `target2`, each
        member is trained on the mean of its t+1 and t+2 (autoregressive) losses and evaluated at t+2.
        Returns the ensemble prediction, acquisition signal (for this rank's first k samples kept in the
        global batch) and the member-averaged loss.
        """
        n_local = state.shape[0]
        state, target, physics, target2 = (self._gather(t) for t in (state, target, physics, target2))
        keep = slice(0, batch_size or state.shape[0])
        state, target = state[keep], target[keep]
        physics = physics[keep] if physics is not None else None
        target2 = target2[keep] if target2 is not None else None
        grid = grid.expand(state.shape[0], *grid.shape[1:])

        preds, losses = [], []
        for model, opt, clip, sched in zip(self.models, self.optimizers, self.clippers, self.schedulers or [None] * len(self.models)):
            pred = model(state, grid, physics)
            loss = criterion(pred, target)
            if target2 is not None:
                pred = model(pred, grid, physics)
                loss = 0.5 * (loss + criterion(pred, target2))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip(list(model.parameters()), step)
            opt.step()
            if sched is not None:
                sched.step()
            preds.append(pred.detach())
            losses.append(loss.detach())

        preds = self._gather(torch.stack(preds))
        loss = self._gather(torch.stack(losses)).mean()
        final_target = target2 if target2 is not None else target
        signal = ensemble_crps(preds, final_target) if self.acquisition_signal == "ecrps" else disagreement(preds)
        mine = slice(self.rank * n_local, min((self.rank + 1) * n_local, state.shape[0]))
        return self._select(preds)[mine], signal[mine], loss

    def forward(self, state, grid, physics=None):
        if self.world == 1 and self.prediction_mode == "member0":
            return self.models[0](state, grid, physics)
        return self._select(self._all_predictions(state, grid, physics))

    def forward_with_uncertainty(self, state, grid, physics=None):
        preds = self._all_predictions(state, grid, physics)
        return self._select(preds), disagreement(preds)

    def _local_key(self, key: str) -> str:
        head, idx, tail = key.split(".", 2)
        return f"{head}.{self.member_ids.index(int(idx))}.{tail}"

    def state_dict(self, *args, optimizers: bool = True, **kwargs):
        local = super().state_dict(*args, **kwargs)
        state = {f"models.{self.member_ids[int(k.split('.', 2)[1])]}.{k.split('.', 2)[2]}": v for k, v in local.items()}
        if optimizers:
            state["_ensemble_optimizer_states"] = {i: o.state_dict() for i, o in zip(self.member_ids, self.optimizers)}
            state["_ensemble_scheduler_states"] = {i: s.state_dict() for i, s in zip(self.member_ids, self.schedulers)}
            state["_ensemble_clip_ema"] = {i: c.ema for i, c in zip(self.member_ids, self.clippers)}
        return state

    def full_state_dict(self, optimizers: bool = True) -> Optional[dict]:
        state = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in self.state_dict(optimizers=optimizers).items()}
        if self.world == 1:
            return state
        parts = [None] * self.world if self.rank == 0 else None
        dist.gather_object(state, parts, dst=dist.get_global_rank(self.group, 0), group=self.group)
        if self.rank != 0:
            return None
        full = {}
        for part in parts:
            for k, v in part.items():
                full[k] = {**full.get(k, {}), **v} if k.startswith("_ensemble") else v
        return full

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = dict(state_dict)
        extras = {k: state_dict.pop(k, None) for k in [k for k in state_dict if k.startswith("_ensemble")]}
        local = {self._local_key(k): v for k, v in state_dict.items() if int(k.split(".", 2)[1]) in self.member_ids}
        result = super().load_state_dict(local, strict=strict)
        opt_states = extras.get("_ensemble_optimizer_states")
        if isinstance(opt_states, dict):
            for i, opt in zip(self.member_ids, self.optimizers):
                opt.load_state_dict(opt_states[i])
            for i, sched in zip(self.member_ids, self.schedulers):
                sched.load_state_dict(extras["_ensemble_scheduler_states"][i])
            for i, clip in zip(self.member_ids, self.clippers):
                clip.ema = extras["_ensemble_clip_ema"][i]
        return result


def local_member_ids(n_members: int, group=None) -> List[int]:
    world = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None else 0
    if n_members % world:
        raise ValueError(f"{n_members} ensemble members cannot be split evenly over {world} training ranks.")
    return list(range(rank, n_members, world))


def member_seed(seed: int, member: int) -> int:
    return 1000 * int(seed or 0) + member

