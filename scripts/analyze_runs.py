#!/usr/bin/env python3
"""
analyze_runs.py

Place your exported CSVs into the `project/result/` folder (files named like `run-...csv`).
This script aggregates runs by algorithm, computes simple summaries and writes:
 - project/result/analysis_summary.csv
 - project/result/avg_reward_curves.png
 - project/result/avg_steps_curves.png

Usage (from the `project` folder):
  py -3 -m pip install --user pandas matplotlib
  py -3 scripts\analyze_runs.py

"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

root = os.path.abspath(os.path.dirname(__file__) + os.sep + '..')
result_dir = os.path.join(root, 'result')
pattern = os.path.join(result_dir, 'run-*.csv')
files = glob.glob(pattern)
if not files:
    print(f'No CSV files found in {result_dir} matching run-*.csv')
    raise SystemExit(1)

runs = []
for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print('Failed to read', f, e)
        continue
    # Expect columns: algorithm,episodes,alpha,gamma,epsilon,episode,reward,steps,reached
    algorithm = df['algorithm'].iloc[0] if 'algorithm' in df.columns else os.path.basename(f)
    episodes = int(df['episodes'].iloc[0]) if 'episodes' in df.columns else int(df['episode'].max())
    runs.append((algorithm, episodes, df))

# Group runs by algorithm
groups = defaultdict(list)
for algo, eps, df in runs:
    groups[algo].append(df)

summary_rows = []
for algo, dfs in groups.items():
    # find shortest length across runs (align by episodes)
    min_len = min(int(df['episode'].max()) for df in dfs)
    # compute per-episode mean and std for reward and steps
    rewards_mat = np.zeros((len(dfs), min_len))
    steps_mat = np.zeros((len(dfs), min_len))
    for i, df in enumerate(dfs):
        rewards_mat[i, :] = [df.loc[df['episode']==ep, 'reward'].values[0] for ep in range(1, min_len+1)]
        steps_mat[i, :] = [df.loc[df['episode']==ep, 'steps'].values[0] for ep in range(1, min_len+1)]
    avg_rewards = rewards_mat.mean(axis=0).tolist()
    std_rewards = rewards_mat.std(axis=0).tolist()
    avg_steps = steps_mat.mean(axis=0).tolist()
    std_steps = steps_mat.std(axis=0).tolist()
    # final-window statistics (last 20 episodes)
    window = min(20, min_len)
    last_rewards_flat = rewards_mat[:, -window:].ravel()
    last_steps_flat = steps_mat[:, -window:].ravel()
    # success rate last window per run
    last_success_rates = []
    for df in dfs:
        succ = [df.loc[df['episode']==ep,'reached'].values[0] for ep in range(min_len-window+1, min_len+1)]
        last_success_rates.append(sum(succ)/len(succ))

    # compute convergence episode per run (first episode where previous `window` episodes
    # have success_rate >= success_thresh and reward std <= reward_std_thresh)
    success_thresh = 0.75
    reward_std_thresh = 1.0
    converge_eps = []
    for i, df in enumerate(dfs):
        conv_ep = None
        # build per-episode arrays for this run
        r = np.array([df.loc[df['episode']==ep,'reward'].values[0] for ep in range(1, min_len+1)])
        reached = np.array([df.loc[df['episode']==ep,'reached'].values[0] for ep in range(1, min_len+1)])
        for ep in range(window, min_len+1):
            recent_reached = reached[ep-window:ep]
            recent_r = r[ep-window:ep]
            if recent_reached.mean() >= success_thresh and recent_r.std() <= reward_std_thresh:
                conv_ep = ep
                break
        converge_eps.append(conv_ep)
    summary_rows.append({
        'algorithm': algo,
        'runs': len(dfs),
        'episodes_used': min_len,
        'mean_reward_last_window': float(pd.Series(last_rewards_flat).mean()),
        'std_reward_last_window': float(pd.Series(last_rewards_flat).std()),
        'mean_steps_last_window': float(pd.Series(last_steps_flat).mean()),
        'success_rate_last_window_mean': float(sum(last_success_rates)/len(last_success_rates)),
        'mean_convergence_episode': float(pd.Series([e for e in converge_eps if e is not None]).mean()) if any(e is not None for e in converge_eps) else None,
        'std_convergence_episode': float(pd.Series([e for e in converge_eps if e is not None]).std()) if any(e is not None for e in converge_eps) else None,
        'runs_converged': sum(1 for e in converge_eps if e is not None)
    })
    # save average curves for this algorithm to result folder
    out_reward_png = os.path.join(result_dir, f'avg_reward_{algo}.png')
    out_steps_png = os.path.join(result_dir, f'avg_steps_{algo}.png')
    out_reward_shaded = os.path.join(result_dir, f'avg_reward_shaded_{algo}.png')
    out_steps_shaded = os.path.join(result_dir, f'avg_steps_shaded_{algo}.png')

    x = np.arange(1, min_len+1)
    # plain mean plots
    plt.figure(figsize=(8,3))
    plt.plot(x, avg_rewards, label=f'{algo} avg')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.title(f'Avg reward per episode — {algo}')
    plt.tight_layout()
    plt.savefig(out_reward_png)
    plt.close()

    plt.figure(figsize=(8,3))
    plt.plot(x, avg_steps, label=f'{algo} avg', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Avg Steps')
    plt.title(f'Avg steps per episode — {algo}')
    plt.tight_layout()
    plt.savefig(out_steps_png)
    plt.close()

    # shaded mean +/- std plots
    plt.figure(figsize=(8,3))
    plt.plot(x, avg_rewards, label='mean')
    plt.fill_between(x, np.array(avg_rewards)-np.array(std_rewards), np.array(avg_rewards)+np.array(std_rewards), color='C0', alpha=0.2, label='±1 std')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.title(f'Avg reward per episode (mean ± std) — {algo}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_reward_shaded)
    plt.close()

    plt.figure(figsize=(8,3))
    plt.plot(x, avg_steps, label='mean', color='orange')
    plt.fill_between(x, np.array(avg_steps)-np.array(std_steps), np.array(avg_steps)+np.array(std_steps), color='C1', alpha=0.2, label='±1 std')
    plt.xlabel('Episode')
    plt.ylabel('Avg Steps')
    plt.title(f'Avg steps per episode (mean ± std) — {algo}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_steps_shaded)
    plt.close()

# write summary CSV
summary_df = pd.DataFrame(summary_rows)
out_summary = os.path.join(result_dir, 'analysis_summary.csv')
summary_df.to_csv(out_summary, index=False)
print('Wrote summary to', out_summary)
print(summary_df.to_string(index=False))

print('\nSaved per-algorithm PNGs in', result_dir)
print('Done')
