"""A Gymnasium-style single-asset trading environment (no gym dependency).

Observation: the last `window` returns (scaled) plus the current position.
Actions:     0 = short, 1 = flat, 2 = long.
Reward:      position * next return - cost * |position change|, in units of
             the series' typical volatility so rewards are well scaled.
"""
from __future__ import annotations

import numpy as np

ACTIONS = np.array([-1.0, 0.0, 1.0])


def ar_returns(n: int = 6000, phi: float = 0.2, vol: float = 0.01, seed: int = 0) -> np.ndarray:
    """AR(1) returns: weakly predictable, so there is something for an agent to learn."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + vol * rng.standard_normal()
    return r


class TradingEnv:
    def __init__(self, returns: np.ndarray, window: int = 10, cost_bps: float = 1.0, episode_len: int | None = None,
                 seed: int = 0):
        self.r = np.asarray(returns, dtype=np.float64)
        self.window, self.cost = window, cost_bps / 1e4
        self.scale = self.r.std() + 1e-12
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)
        self.obs_dim = window + 1
        self.n_actions = len(ACTIONS)

    def reset(self, start: int | None = None):
        last_start = len(self.r) - 1 - (self.episode_len or (len(self.r) - self.window - 1))
        if start is None:
            start = self.window if self.episode_len is None else int(self.rng.integers(self.window, max(self.window + 1, last_start)))
        self.t = start
        self.end = len(self.r) - 1 if self.episode_len is None else min(len(self.r) - 1, start + self.episode_len)
        self.pos = 0.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        hist = self.r[self.t - self.window : self.t] / self.scale
        return np.append(hist, self.pos).astype(np.float32)

    def step(self, action: int):
        new_pos = ACTIONS[action]
        trade_cost = self.cost * abs(new_pos - self.pos)
        pnl = new_pos * self.r[self.t] - trade_cost  # return earned from t-1 close to t close, decided at t-1
        self.pos = new_pos
        self.t += 1
        done = self.t >= self.end
        return self._obs(), float(pnl / self.scale), done, False, {"pnl": pnl}
