"""Train the prism PPO navigation policy *inside* the learned world model.

    python -m prism.nav.train_rl [--steps 150000]

The environment is `rl_env.WorldModelEnv`, whose transition function is the tiny
latent dynamics model - no video, no simulator, no physics engine.  Initial
states are latents of real cached BEV maps when the `bev` cache exists and of
`bev_utils` stand-in scenes otherwise (printed at the top of every run).

Evaluation is deliberately unforgiving:

* held-out initial states never used for training;
* three baselines - the trained PPO policy, an always-FORWARD policy, and a
  uniform random policy;
* collisions are counted **geometrically** (`bev_utils.geometric_collision`),
  by warping the real BEV map alongside the latent, so the score does not depend
  on the world model's own opinion of its own risk head;
* progress in metres, and the STOP rate, are reported next to the collision rate,
  because a policy that never moves trivially never collides and that is the
  failure mode this project is arguing against.

If PPO does not beat the baselines, the table says so.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..config import CFG, ACTIONS, N_ACTIONS, CKPT_DIR, WORK_DIR
from ..io_utils import set_seed
from ..models.world_model import DrishtiWorldModel, WM_CKPT
from . import bev_utils as bu
from .rl_env import (WorldModelEnv, RewardWeights, collect_init_states, encode_bank,
                     PPO_CKPT, OBS_DIM)

RL_LOG = WORK_DIR / "rl_train_log.npz"


# ---------------------------------------------------------------- evaluation

def eval_policy(env: WorldModelEnv, act_fn, n_episodes: int = 60) -> dict:
    """Run a policy on held-out initial states; return honest aggregate metrics."""
    rets, prog, coll, stops, lens = [], [], [], [], []
    for e in range(n_episodes):
        obs, _ = env.reset(options={"index": e})
        done, R, n_stop, n = False, 0.0, 0, 0
        info = {}
        while not done:
            a = act_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            R += r
            n_stop += int(bu.action_motion(a)[0] < 0.10)
            n += 1
            done = term or trunc
        rets.append(R)
        prog.append(info.get("progress", 0.0))
        coll.append(int(info.get("true_hits", 0) > 0))
        stops.append(n_stop / max(n, 1))
        lens.append(n)
    return {"return": float(np.mean(rets)), "return_std": float(np.std(rets)),
            "progress_m": float(np.mean(prog)),
            "collision_rate": float(np.mean(coll)),
            "stop_rate": float(np.mean(stops)),
            "ep_len": float(np.mean(lens))}


def _fmt_row(name, m):
    return (f"  {name:22s} return {m['return']:+8.2f} +- {m['return_std']:5.2f} | "
            f"progress {m['progress_m']:6.2f} m | collisions {m['collision_rate']*100:5.1f}% | "
            f"stop-rate {m['stop_rate']*100:5.1f}% | ep len {m['ep_len']:4.1f}")


# ---------------------------------------------------------------- training

def train(steps: int = 150_000, n_envs: int = 8, seed: int = CFG.seed,
          device: str = "cpu", eval_episodes: int = 60) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback

    set_seed(seed)
    wm = DrishtiWorldModel()
    if WM_CKPT.exists():
        sd = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
        wm.load_state_dict(sd["model"] if "model" in sd else sd)
        wm_src = str(WM_CKPT)
    else:
        wm_src = "UNTRAINED world model (checkpoints/world_model.pt missing)"
    wm.eval()
    print(f"[world model] {wm_src}  ({wm.param_counts()['total']:,} params)")

    states, src = collect_init_states()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(states.shape[0])
    n_hold = max(8, int(0.2 * len(perm)))
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    tr_states, ho_states = states[train_idx], states[hold_idx]
    print(f"[init states] {states.shape[0]} scenes from {src}")
    print(f"[split] {len(train_idx)} training / {len(hold_idx)} held-out initial states")

    lat_tr, ctx_tr = encode_bank(wm, tr_states, device)
    lat_ho, ctx_ho = encode_bank(wm, ho_states, device)

    def mk(i):
        def _f():
            return Monitor(WorldModelEnv(wm, lat_tr, ctx_tr, None, device=device,
                                         seed=seed + i))
        return _f

    venv = DummyVecEnv([mk(i) for i in range(n_envs)])
    model = PPO("MlpPolicy", venv, device="cpu", seed=seed,
                n_steps=256, batch_size=256, n_epochs=8, gamma=0.98,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.008,
                learning_rate=3e-4, verbose=0,
                policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]))
    n_pol = sum(p.numel() for p in model.policy.parameters())
    print(f"[ppo] obs_dim {OBS_DIM}, {N_ACTIONS} actions, policy {n_pol:,} params, "
          f"{n_envs} parallel envs, target {steps:,} steps")

    curve = {"t": [], "rew": [], "len": []}

    class Curve(BaseCallback):
        def _on_rollout_end(self):
            buf = self.model.ep_info_buffer
            if buf:
                curve["t"].append(self.num_timesteps)
                curve["rew"].append(float(np.mean([e["r"] for e in buf])))
                curve["len"].append(float(np.mean([e["l"] for e in buf])))
                if len(curve["t"]) % 20 == 0:
                    print(f"    {self.num_timesteps:>8,} steps  ep_rew_mean "
                          f"{curve['rew'][-1]:+7.2f}  ep_len {curve['len'][-1]:.1f}")

        def _on_step(self):
            return True

    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, callback=Curve(), progress_bar=False)
    train_s = time.perf_counter() - t0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(PPO_CKPT))
    print(f"[ppo] trained {steps:,} steps in {train_s:.0f}s -> {PPO_CKPT}")

    # ------------------------------------------------------------- evaluate
    ev_env = WorldModelEnv(wm, lat_ho, ctx_ho, ho_states, device=device,
                           track_map=True, seed=seed + 991)
    n_ep = min(eval_episodes, lat_ho.shape[0])

    def ppo_act(o):
        a, _ = model.predict(o, deterministic=True)
        return int(a)

    rrng = np.random.default_rng(7)
    results = {
        "ppo": eval_policy(ev_env, ppo_act, n_ep),
        "always_forward": eval_policy(ev_env, lambda o: 0, n_ep),
        "random": eval_policy(ev_env, lambda o: int(rrng.integers(0, N_ACTIONS)), n_ep),
        "always_stop": eval_policy(ev_env, lambda o: ACTIONS.index("STOP"), n_ep),
    }

    print(f"\n[eval] {n_ep} held-out initial states, {ev_env.max_steps} steps "
          f"({ev_env.max_steps*bu.WM_DT:.1f} s) each")
    print("  collisions are GEOMETRIC (footprint vs the warped real BEV map), "
          "not the world model's own risk head")
    for k in ("ppo", "always_forward", "random", "always_stop"):
        print(_fmt_row(k, results[k]))

    p, f, r = results["ppo"], results["always_forward"], results["random"]
    verdict = []

    # Lead with safety, not with return. Return is the quantity the policy was trained to
    # maximise *inside the learned world model*; the geometric collision rate is the
    # held-out quantity it never saw and cannot game. Reporting return first would let a
    # policy that exploits world-model error read as a success.
    exploits = p["collision_rate"] > f["collision_rate"] + 0.05
    verdict.append(f"geometric collision rate - PPO {p['collision_rate']*100:.1f}%, "
                   f"always-FORWARD {f['collision_rate']*100:.1f}%, "
                   f"random {r['collision_rate']*100:.1f}%, "
                   f"always-STOP {results['always_stop']['collision_rate']*100:.1f}%")
    if exploits:
        verdict.append(
            "HEADLINE: the PPO policy is NOT safer than driving blindly forward. It "
            "collides more often on the geometric check while scoring a higher return, "
            "which means it has learned to exploit errors in the learned world model "
            "rather than to avoid real obstacles. Collision is scored against the warped "
            "real BEV map and is deliberately kept out of the reward, so this is a "
            "held-out measurement of world-model error, not a reward bug.")
        verdict.append(
            "CONSEQUENCE: this is exactly why the policy never drives the vehicle. PPO "
            "only proposes an action; the safety supervisor gates it on terrain step, "
            "confidence, unknown fraction and stopping space. The gap between these two "
            "numbers is the argument for that supervisor, and it is reported, not hidden.")
    else:
        verdict.append(f"PPO is at least as safe as always-FORWARD on the held-out "
                       f"geometric check ({p['collision_rate']*100:.1f}% vs "
                       f"{f['collision_rate']*100:.1f}%)")
    verdict.append(f"progress - PPO {p['progress_m']:.2f} m, always-FORWARD "
                   f"{f['progress_m']:.2f} m, random {r['progress_m']:.2f} m, "
                   f"at a {p['stop_rate']*100:.0f}% PPO stop rate")
    verdict.append(f"in-model return (the trained objective) - PPO {p['return']:+.2f}, "
                   f"always-FORWARD {f['return']:+.2f}, random {r['return']:+.2f}. "
                   f"In-model return is not evidence of real-world safety.")
    if p["stop_rate"] > 0.85:
        verdict.append("WARNING: the policy has collapsed to standing still - "
                       "it is safe and useless, which is exactly the failure mode "
                       "the reward's idle penalty is meant to prevent")
    print("\n[verdict]")
    for v in verdict:
        print("  " + v)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"steps": steps, "n_envs": n_envs, "train_seconds": train_s,
            "world_model": wm_src, "init_state_source": src,
            "n_train_states": int(len(train_idx)), "n_holdout_states": int(len(hold_idx)),
            "policy_params": int(n_pol), "eval": results, "verdict": verdict,
            "reward_weights": RewardWeights().__dict__}
    np.savez_compressed(RL_LOG,
                        curve_t=np.asarray(curve["t"], np.float32),
                        curve_rew=np.asarray(curve["rew"], np.float32),
                        curve_len=np.asarray(curve["len"], np.float32),
                        meta=np.array(json.dumps(meta)))
    print(f"\n[save] {PPO_CKPT}\n[save] {RL_LOG}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=60)
    a = ap.parse_args()
    train(a.steps, a.envs, eval_episodes=a.eval_episodes)
