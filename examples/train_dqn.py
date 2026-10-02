"""Train on the first 70% of an AR(1) series, evaluate greedily on the unseen 30%."""
import numpy as np

from rlta import DQNAgent, TradingEnv, ar_returns, evaluate, train

r = ar_returns(n=8000, phi=0.2, seed=0)
split = int(0.7 * len(r))
train_env = TradingEnv(r[:split], window=10, cost_bps=1.0, episode_len=500, seed=0)
test_env = TradingEnv(r[split:], window=10, cost_bps=1.0)

agent = DQNAgent(train_env.obs_dim, train_env.n_actions, seed=0)
train(train_env, agent, episodes=60)

def sharpe(p):
    return p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0.0

pnl = evaluate(test_env, agent)
test_r = r[split + test_env.window : split + test_env.window + len(pnl)]
oracle = np.sign(0.2 * np.r_[0, test_r[:-1]]) * test_r  # trades the true AR coefficient, no costs
print(f"DQN (out of sample, after costs) Sharpe: {sharpe(pnl):.2f}")
print(f"buy & hold Sharpe:                      {sharpe(test_r):.2f}")
print(f"oracle AR(1) rule Sharpe (no costs):     {sharpe(oracle):.2f}")
