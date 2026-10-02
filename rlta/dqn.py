"""Double DQN with a target network and experience replay."""
from __future__ import annotations

import copy
import random
from collections import deque

import numpy as np
import torch
from torch import nn


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3, gamma: float = 0.9, buffer: int = 50_000,
                 batch: int = 128, target_sync: int = 500, seed: int = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        self.q = QNet(obs_dim, n_actions)
        self.target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.batch, self.target_sync = gamma, batch, target_sync
        self.memory: deque = deque(maxlen=buffer)
        self.n_actions, self.steps = n_actions, 0

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            return int(self.q(torch.from_numpy(obs)).argmax())

    def remember(self, *transition) -> None:
        self.memory.append(transition)

    def learn(self) -> float | None:
        if len(self.memory) < self.batch:
            return None
        s, a, r, s2, d = map(np.array, zip(*random.sample(self.memory, self.batch)))
        s, s2 = torch.from_numpy(s), torch.from_numpy(s2)
        a, r, d = torch.from_numpy(a).long(), torch.from_numpy(r).float(), torch.from_numpy(d).float()
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():  # double DQN: online net picks the action, target net scores it
            best = self.q(s2).argmax(1, keepdim=True)
            target = r + self.gamma * (1 - d) * self.target(s2).gather(1, best).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.detach())


def train(env, agent: DQNAgent, episodes: int = 60, eps_start: float = 1.0, eps_end: float = 0.05) -> list[float]:
    history = []
    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - ep / (0.7 * episodes))
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            a = agent.act(obs, eps)
            obs2, r, done, _, info = env.step(a)
            agent.remember(obs, a, r, obs2, done)
            agent.learn()
            obs, total = obs2, total + info["pnl"]
        history.append(total)
    return history


def evaluate(env, agent: DQNAgent) -> np.ndarray:
    """Greedy rollout over the whole series; returns the per-step P&L."""
    obs, _ = env.reset(start=env.window)
    pnl, done = [], False
    while not done:
        obs, _, done, _, info = env.step(agent.act(obs, 0.0))
        pnl.append(info["pnl"])
    return np.asarray(pnl)
