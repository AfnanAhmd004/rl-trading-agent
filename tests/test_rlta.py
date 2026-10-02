import numpy as np
import pytest

from rlta import DQNAgent, TradingEnv, ar_returns, evaluate, train


def test_env_reward_and_costs():
    r = np.array([0.0] * 5 + [0.01, -0.02, 0.03])
    env = TradingEnv(r, window=5, cost_bps=10)
    obs, _ = env.reset(start=5)
    assert obs.shape == (6,) and obs[-1] == 0
    _, _, _, _, info = env.step(2)  # go long: earn r[5]=0.01, pay 10 bps on a size-1 trade
    assert info["pnl"] == pytest.approx(0.01 - 0.001)
    _, _, _, _, info = env.step(0)  # flip to short: earn -(-0.02), pay for a size-2 trade
    assert info["pnl"] == pytest.approx(0.02 - 0.002)


def test_observation_has_no_future():
    r = ar_returns(200, seed=1)
    env = TradingEnv(r, window=10)
    obs, _ = env.reset(start=50)
    assert np.allclose(obs[:-1] * env.scale, r[40:50])


def test_dqn_learns_predictable_series():
    r = ar_returns(4000, phi=0.5, seed=2)
    env = TradingEnv(r[:3000], window=5, cost_bps=0, episode_len=300, seed=0)
    agent = DQNAgent(env.obs_dim, env.n_actions, seed=0)
    train(env, agent, episodes=25)
    pnl = evaluate(TradingEnv(r[3000:], window=5, cost_bps=0), agent)
    assert pnl.mean() > 0
