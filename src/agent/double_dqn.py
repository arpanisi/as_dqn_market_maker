"""Step 6 PyTorch Double-DQN implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from settings import DEFAULT_SETTINGS


@dataclass(frozen=True)
class Action:
    index: int
    risk_aversion: float
    skew: float


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


def action_space(
    risk_aversions: Iterable[float] | None = None,
    skews: Iterable[float] | None = None,
) -> tuple[Action, ...]:
    gammas = tuple(risk_aversions or DEFAULT_SETTINGS.agent.risk_aversions)
    skew_values = tuple(skews or DEFAULT_SETTINGS.agent.skews)
    return tuple(
        Action(index, float(gamma), float(skew))
        for index, (gamma, skew) in enumerate((gamma, skew) for gamma in gammas for skew in skew_values)
    )


def build_state(
    *,
    inventory_q: float,
    realized_volatility: float,
    arrival_A: float,
    arrival_kappa: float,
    time_remaining_fraction: float,
    adverse_selection: float,
    funding_rate: float,
) -> np.ndarray:
    return np.array(
        [
            float(inventory_q),
            float(realized_volatility),
            float(arrival_A),
            float(arrival_kappa),
            float(np.clip(time_remaining_fraction, 0.0, 1.0)),
            float(adverse_selection),
            float(funding_rate) / 0.01,
        ],
        dtype=np.float32,
    )


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self._items: list[Transition] = []
        self._pos = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: Transition) -> None:
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        idx = self._rng.choice(len(self._items), size=int(batch_size), replace=False)
        return [self._items[int(i)] for i in idx]


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_units: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class DoubleDQNAgent:
    def __init__(
        self,
        state_dim: int = 7,
        seed: int = 0,
        warmup_transitions: int | None = None,
        batch_size: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        cfg = DEFAULT_SETTINGS.agent
        torch.manual_seed(seed)
        self.actions = action_space()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(state_dim, len(self.actions), cfg.hidden_units).to(self.device)
        self.target = QNetwork(state_dim, len(self.actions), cfg.hidden_units).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.replay = ReplayBuffer(cfg.replay_capacity, seed)
        self.warmup_transitions = int(warmup_transitions or cfg.warmup_transitions)
        self.batch_size = int(batch_size or cfg.batch_size)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.gradient_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> Action:
        if self.rng.random() < epsilon:
            return self.actions[int(self.rng.integers(0, len(self.actions)))]
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action_idx = int(self.online(state_tensor).argmax(dim=1).item())
        return self.actions[action_idx]

    def epsilon_for_step(self, step: int, total_training_steps: int) -> float:
        cfg = DEFAULT_SETTINGS.agent
        decay_steps = max(1, int(total_training_steps * cfg.epsilon_decay_fraction))
        fraction = min(1.0, max(0.0, step / decay_steps))
        return float(cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start))

    def learn_from_transition(self, transition: Transition) -> float | None:
        cfg = DEFAULT_SETTINGS.agent
        self.replay.add(transition)
        self.steps += 1
        if len(self.replay) < self.warmup_transitions:
            return None

        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.bool, device=self.device)

        q_values = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + (~dones).float() * cfg.discount_beta * next_q

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % cfg.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "steps": self.steps,
                "gradient_steps": self.gradient_steps,
            },
            output,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(checkpoint["target_state_dict"])
        self.steps = int(checkpoint.get("steps", 0))
        self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
