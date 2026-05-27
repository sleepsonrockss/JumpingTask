#!/usr/bin/env python3
"""
Monte Carlo (First-Visit), SARSA, and Q-Learning agents for the Jumping Task.

State representation
--------------------
With finish_jump=True the jump executes atomically: the agent only decides
*when* to initiate a jump.  The one feature that drives that decision is
how far the agent is from the obstacle:

    state = (dist_to_obstacle,)   where dist = obstacle_x - agent_x

This generalises across all obstacle positions and floor heights (the jump
arc is the same regardless of floor height).

Usage
-----
    python3 rl_agents.py              # train all three agents then run visual demo
    python3 rl_agents.py --no-render  # train only, save plot, skip pygame window
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # headless-safe; flip to "TkAgg" / "MacOSX" if needed
import matplotlib.pyplot as plt
import numpy as np

# ── import the environment ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gym_jumping_task.envs.jumping_task import JumpTaskEnv

# ── constants ─────────────────────────────────────────────────────────────────
DIST_CLIP_LO = -10   # agent well past the obstacle
DIST_CLIP_HI =  60   # agent far to the left of obstacle
N_ACTIONS    =   2   # 0 = right, 1 = jump


# ── state extraction ──────────────────────────────────────────────────────────

def get_state(env: JumpTaskEnv) -> tuple:
    dist = int(env.obstacle_position - env.agent_pos_x)
    dist = max(DIST_CLIP_LO, min(dist, DIST_CLIP_HI))
    return (dist,)


# ── exploration policy ────────────────────────────────────────────────────────

def eps_greedy(Q: dict, state: tuple, eps: float) -> int:
    if np.random.random() < eps:
        return np.random.randint(N_ACTIONS)
    return int(np.argmax(Q[state]))


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo – First-Visit
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo(
    env,
    n_episodes: int = 3000,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_min: float = 0.05,
) -> tuple:
    """First-Visit Monte Carlo control with epsilon-greedy exploration."""
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    returns_sum   = defaultdict(float)
    returns_count = defaultdict(int)
    rewards_log, success_log = [], []
    eps = eps_start
    eps_decay = (eps_min / eps_start) ** (1.0 / n_episodes)

    print("Training Monte Carlo (First-Visit)…")
    for ep in range(n_episodes):
        env.reset()
        s = get_state(env)
        trajectory, done = [], False

        # ── collect episode ──────────────────────────────────────────────────
        while not done:
            a = eps_greedy(Q, s, eps)
            _, r, done, info = env.step(a)
            trajectory.append((s, a, r))
            s = get_state(env)

        eps = max(eps_min, eps * eps_decay)

        # ── first-visit return update ────────────────────────────────────────
        G, visited = 0.0, set()
        for s_t, a_t, r_t in reversed(trajectory):
            G = gamma * G + r_t
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns_sum[(s_t, a_t)]   += G
                returns_count[(s_t, a_t)] += 1
                Q[s_t][a_t] = returns_sum[(s_t, a_t)] / returns_count[(s_t, a_t)]

        total_r = sum(r for _, _, r in trajectory)
        success = _episode_success(env, info)
        rewards_log.append(total_r)
        success_log.append(int(success))

        if (ep + 1) % 500 == 0:
            _log("MC     ", ep + 1, n_episodes, rewards_log, success_log)

    return Q, rewards_log, success_log


# ─────────────────────────────────────────────────────────────────────────────
# SARSA – on-policy TD(0)
# ─────────────────────────────────────────────────────────────────────────────

def sarsa(
    env,
    n_episodes: int = 3000,
    alpha: float  = 0.1,
    gamma: float  = 0.99,
    eps_start: float = 1.0,
    eps_min: float   = 0.05,
) -> tuple:
    """On-policy SARSA with epsilon-greedy exploration."""
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    rewards_log, success_log = [], []
    eps = eps_start
    eps_decay = (eps_min / eps_start) ** (1.0 / n_episodes)

    print("Training SARSA…")
    for ep in range(n_episodes):
        env.reset()
        s = get_state(env)
        a = eps_greedy(Q, s, eps)
        total_r, done, last_info = 0.0, False, {}

        while not done:
            _, r, done, info = env.step(a)
            s2 = get_state(env)
            a2 = eps_greedy(Q, s2, eps)

            # SARSA update: uses next *behaviour* action a2
            td_target = r + gamma * Q[s2][a2] * (not done)
            Q[s][a]  += alpha * (td_target - Q[s][a])

            s, a, total_r, last_info = s2, a2, total_r + r, info

        eps = max(eps_min, eps * eps_decay)
        rewards_log.append(total_r)
        success_log.append(int(_episode_success(env, last_info)))

        if (ep + 1) % 500 == 0:
            _log("SARSA  ", ep + 1, n_episodes, rewards_log, success_log)

    return Q, rewards_log, success_log


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning – off-policy TD(0)
# ─────────────────────────────────────────────────────────────────────────────

def q_learning(
    env,
    n_episodes: int = 3000,
    alpha: float  = 0.1,
    gamma: float  = 0.99,
    eps_start: float = 1.0,
    eps_min: float   = 0.05,
) -> tuple:
    """Off-policy Q-Learning with epsilon-greedy exploration."""
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    rewards_log, success_log = [], []
    eps = eps_start
    eps_decay = (eps_min / eps_start) ** (1.0 / n_episodes)

    print("Training Q-Learning…")
    for ep in range(n_episodes):
        env.reset()
        s = get_state(env)
        total_r, done, last_info = 0.0, False, {}

        while not done:
            a = eps_greedy(Q, s, eps)
            _, r, done, info = env.step(a)
            s2 = get_state(env)

            # Q-Learning update: uses greedy max over next state (off-policy)
            td_target = r + gamma * np.max(Q[s2]) * (not done)
            Q[s][a]  += alpha * (td_target - Q[s][a])

            s, total_r, last_info = s2, total_r + r, info

        eps = max(eps_min, eps * eps_decay)
        rewards_log.append(total_r)
        success_log.append(int(_episode_success(env, last_info)))

        if (ep + 1) % 500 == 0:
            _log("Q-Learn", ep + 1, n_episodes, rewards_log, success_log)

    return Q, rewards_log, success_log


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _episode_success(env: JumpTaskEnv, info: dict) -> bool:
    """True iff the agent cleared the screen (not a collision or timeout)."""
    return (
        env.done
        and not info.get("collision", False)
        and env.agent_pos_x + env.agent_size[0] > env.scr_w
    )


def _log(tag: str, ep: int, total: int, rewards: list, successes: list):
    n   = min(500, len(rewards))
    avg_r = np.mean(rewards[-n:])
    avg_s = np.mean(successes[-n:]) * 100
    print(f"  [{tag}] Ep {ep:4d}/{total}  |  "
          f"Avg Reward: {avg_r:7.1f}  |  Success: {avg_s:5.1f}%")


def evaluate(Q: dict, env: JumpTaskEnv, n_episodes: int = 200) -> tuple:
    """Greedy evaluation; returns (mean_reward, success_rate_pct)."""
    rews, succs = [], []
    for _ in range(n_episodes):
        env.reset()
        s = get_state(env)
        total, done, last_info = 0.0, False, {}
        while not done:
            a = int(np.argmax(Q[s]))
            _, r, done, info = env.step(a)
            s, total, last_info = get_state(env), total + r, info
        rews.append(total)
        succs.append(int(_episode_success(env, last_info)))
    return float(np.mean(rews)), float(np.mean(succs) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(arr, w: int = 50) -> np.ndarray:
    if len(arr) < w:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_results(results: dict, save_path: str):
    colors = {
        "Monte Carlo": "#1f77b4",
        "SARSA":       "#ff7f0e",
        "Q-Learning":  "#2ca02c",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RL Agents – Jumping Task", fontsize=13)

    for name, (rews, succs) in results.items():
        c = colors[name]
        ax1.plot(_smooth(rews),                          label=name, color=c, lw=1.8)
        ax2.plot(_smooth(np.array(succs, float)) * 100,  label=name, color=c, lw=1.8)

    ax1.set(title="Episode Reward (smoothed w=50)", xlabel="Episode", ylabel="Total Reward")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set(title="Success Rate (smoothed w=50)", xlabel="Episode", ylabel="Success Rate (%)")
    ax2.set_ylim(0, 110); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nTraining curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Visual demo
# ─────────────────────────────────────────────────────────────────────────────

def run_visual_demo(policies: dict, n_demos: int = 3):
    """
    Launch a single pygame window and watch each trained agent play in turn.
    policies: ordered dict of {algo_name: Q_table}
    """
    try:
        import pygame
        pygame.init()
    except Exception as exc:
        print(f"  pygame unavailable ({exc}), skipping visual demo.")
        return

    print(f"\n{'='*55}")
    print(f"Visual demo: all three agents, {n_demos} episodes each")
    print("Close the window or press Ctrl-C to stop early.")
    print("="*55)

    env = JumpTaskEnv(
        rendering=True,
        zoom=8,
        slow_motion=True,   # 0.1 s per step → human-watchable speed
        finish_jump=True,   # jump animates frame-by-frame via _game_status
    )

    try:
        for algo_name, Q in policies.items():
            print(f"\n  --- {algo_name} ---")
            for ep in range(n_demos):
                env.reset()
                env.render()
                # Update window title so the user sees which agent is playing
                pygame.display.set_caption(f"{algo_name}  —  episode {ep+1}/{n_demos}")
                s    = get_state(env)
                total, done = 0.0, False

                while not done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt

                    a = int(np.argmax(Q[s]))
                    _, r, done, info = env.step(a)
                    s = get_state(env)
                    total += r

                ok = "SUCCESS ✓" if _episode_success(env, info) else "FAILED  ✗"
                print(f"    Ep {ep+1}/{n_demos}: {ok}  (reward={total:.1f})")
                time.sleep(0.6)

    except KeyboardInterrupt:
        print("  Demo interrupted.")
    finally:
        env.close()
        try:
            pygame.quit()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MC / SARSA / Q-Learning on the Jumping Task")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Number of training episodes per algorithm (default: 3000)")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip the pygame visual demo after training")
    args = parser.parse_args()

    N = args.episodes

    # Shared training environment (no rendering → fast)
    env = JumpTaskEnv(finish_jump=True)

    results  = {}   # name → (rewards_log, success_log)
    policies = {}   # name → Q

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print("=" * 55)
    Q_mc, r_mc, s_mc = monte_carlo(env, n_episodes=N)
    avg_r, avg_s = evaluate(Q_mc, env)
    print(f"  → Final eval (200 eps): reward={avg_r:.1f}  success={avg_s:.1f}%")
    results["Monte Carlo"]  = (r_mc, s_mc)
    policies["Monte Carlo"] = Q_mc

    # ── SARSA ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    Q_sarsa, r_sarsa, s_sarsa = sarsa(env, n_episodes=N)
    avg_r, avg_s = evaluate(Q_sarsa, env)
    print(f"  → Final eval (200 eps): reward={avg_r:.1f}  success={avg_s:.1f}%")
    results["SARSA"]  = (r_sarsa, s_sarsa)
    policies["SARSA"] = Q_sarsa

    # ── Q-Learning ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    Q_ql, r_ql, s_ql = q_learning(env, n_episodes=N)
    avg_r, avg_s = evaluate(Q_ql, env)
    print(f"  → Final eval (200 eps): reward={avg_r:.1f}  success={avg_s:.1f}%")
    results["Q-Learning"]  = (r_ql, s_ql)
    policies["Q-Learning"] = Q_ql

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_results.png")
    plot_results(results, plot_path)

    # ── Print final Q-table snapshot ──────────────────────────────────────────
    print("\nQ-Learning policy (dist_to_obstacle → preferred action):")
    print(f"  {'dist':>6}  {'Q[right]':>10}  {'Q[jump]':>10}  {'choice':>8}")
    for d in range(30, -11, -1):
        s = (d,)
        q = policies["Q-Learning"][s]
        choice = "JUMP" if np.argmax(q) == 1 else "right"
        if q[0] != 0 or q[1] != 0:   # only print non-zero entries
            print(f"  {d:>6}  {q[0]:>10.2f}  {q[1]:>10.2f}  {choice:>8}")

    # ── Visual demo ───────────────────────────────────────────────────────────
    if not args.no_render:
        run_visual_demo(policies, n_demos=3)

    print("\nAll done.")


if __name__ == "__main__":
    main()
