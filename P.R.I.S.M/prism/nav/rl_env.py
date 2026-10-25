"""Gymnasium environment whose dynamics ARE the learned world model.

That is the whole point of the prediction stage: once the BEV world state is
compressed into a 96-D latent and a GRU can advance that latent under a candidate
action, the world model *is* the simulator.  No video, no renderer, no physics
engine - a PPO rollout step is one GRU cell plus three tiny heads, so millions of
simulated steps cost minutes on a CPU.

    observation : [ latent (96) | scene context (3) | dynamic (3) | last action (6) ]
                  = 108 floats, all roughly in [-1, 1]
    action      : Discrete(6) - the prism action set
    dynamics    : s' = LatentDynamics(s, a)              (learned)
    reward      : progress - risk - unknown - jerk - idling
    termination : the world model's collision-risk head exceeds
                  `CFG.safety.risk_stop`, or the episode length runs out

Reward design (and why the idling penalty matters)
--------------------------------------------------
A navigation policy that outputs STOP everywhere has zero collisions and is
completely useless - it fails the mission rather than the safety test.  That
trade-off is the central argument of this project, so the reward states it
explicitly: forward progress is rewarded, predicted collision risk and entering
low-traversability / unknown territory are penalised, and there is a standing
penalty for not moving.  A policy is only good if it finds speed *and* safety;
the supervisor then remains as the hard gate on top.

Initial states are encoded from **real cached BEV maps** where the `bev` cache
exists, so the latents PPO trains on are latents of scenes that actually occurred
in the five clips.  Where the cache is not there yet, `bev_utils.synthetic_state`
stand-ins are used and this is reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from ..config import CFG, ACTIONS, N_ACTIONS, CKPT_DIR
from ..types import FramePacket
from . import bev_utils as bu
from ..models.world_model import DrishtiWorldModel, WM_CKPT

PPO_CKPT = CKPT_DIR / "ppo_drishti.zip"

# The env steps a batch-of-1 GRU millions of times.  With torch's default thread
# pool that is ~28 ms/step on this (heavily shared) machine; pinned to one thread
# it is ~1.4 ms.  Measured, not assumed - see the self-test at the bottom.
try:
    torch.set_num_threads(1)
except Exception:                                    # pragma: no cover
    pass

OBS_LATENT = CFG.wm.state_dim
OBS_CTX = 3          # scene mean confidence, unknown fraction, initial traversability
OBS_DYN = 3          # predicted traversability, predicted risk, heading error
OBS_LAST = N_ACTIONS
OBS_DIM = OBS_LATENT + OBS_CTX + OBS_DYN + OBS_LAST + 1     # + normalised episode time


@dataclass
class RewardWeights:
    """Reward shaping.

    One property matters more than the individual numbers: **a safe forward step
    must score positively**.  In an earlier version every term was a penalty and
    the per-step reward of even a clean forward step was negative, so the fastest
    way to a high return was to trigger the terminal condition early and stop
    accruing negatives.  PPO duly learned to crash or freeze within six steps and
    still "beat" always-FORWARD on return - a reward bug wearing the costume of a
    result.  The `alive` bonus plus a progress term larger than the typical risk
    penalty fixes the sign; the risk and unknown terms only bite above a
    deadband, so an ordinary clear corridor costs nothing.
    """
    progress: float = 4.0        # per metre advanced along the goal direction
    alive: float = 0.15          # per step survived, so early termination is bad
    risk: float = 2.5            # per unit of predicted collision risk above the deadband
    risk_deadband: float = 0.15
    unknown: float = 0.8         # per unit of predicted non-traversability below trav_ok
    trav_ok: float = 0.60
    jerk: float = 0.06           # per action switch
    idle: float = 0.50           # per step spent not moving
    collision: float = 10.0      # terminal penalty
    heading: float = 0.25        # per radian of heading error from the goal


class WorldModelEnv(gym.Env):
    """PPO trains here.  One env step = one `LatentDynamics` step of `WM_DT` seconds."""

    metadata = {"render_modes": []}

    def __init__(self,
                 wm: DrishtiWorldModel,
                 init_latents: np.ndarray,
                 init_ctx: np.ndarray,
                 init_maps: Optional[np.ndarray] = None,
                 max_steps: int = 32,
                 weights: Optional[RewardWeights] = None,
                 device: str = "cpu",
                 seed: int = CFG.seed,
                 track_map: bool = False):
        super().__init__()
        self.device = torch.device(device)
        self.wm = wm.to(self.device).eval()
        for p in self.wm.parameters():
            p.requires_grad_(False)
        self.lat = np.asarray(init_latents, np.float32)
        self.ctx = np.asarray(init_ctx, np.float32)
        self.maps = init_maps
        self.max_steps = int(max_steps)
        self.w = weights or RewardWeights()
        self.track_map = bool(track_map and init_maps is not None)

        self.observation_space = spaces.Box(-4.0, 4.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._rng = np.random.default_rng(seed)
        self._s = None
        self._hs = None
        self._reset_internal(0)

    # ------------------------------------------------------------------
    def _reset_internal(self, idx: int):
        self._idx = int(idx)
        self._s = torch.from_numpy(self.lat[idx:idx + 1].copy()).to(self.device)
        self._hs = self.wm.dynamics.init_hidden(1, self.device)
        self._ctx = self.ctx[idx].copy()
        self._t = 0
        self._yaw = 0.0
        self._dist = 0.0
        self._last_a = ACTIONS.index("STOP")
        self._trav = float(self._ctx[2])
        self._risk = 0.0
        self._map = (self.maps[idx].astype(np.float32).copy()
                     if self.track_map else None)
        self._true_risk = 0.0
        self._true_hits = 0

    def _obs(self) -> np.ndarray:
        last = np.zeros(N_ACTIONS, np.float32)
        last[self._last_a] = 1.0
        return np.concatenate([
            self._s.detach().cpu().numpy().ravel(),
            self._ctx.astype(np.float32),
            np.array([self._trav, self._risk,
                      np.clip(self._yaw / np.pi, -1, 1)], np.float32),
            last,
            np.array([self._t / self.max_steps], np.float32),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(0, self.lat.shape[0]))
        if options and "index" in options:
            idx = int(options["index"]) % self.lat.shape[0]
        self._reset_internal(idx)
        return self._obs(), {"init_index": idx}

    @torch.no_grad()
    def step(self, action: int):
        a = int(action)
        v, w = bu.action_motion(a)
        dt = bu.WM_DT

        a1h = torch.zeros(1, N_ACTIONS, device=self.device)
        a1h[0, a] = 1.0
        self._s, self._hs = self.wm.dynamics.step(self._s, a1h, self._hs)
        occ, trav, risk = self.wm.decode(self._s)
        self._trav = float(trav.item())
        self._risk = float(risk.item())

        # pose bookkeeping so "progress toward the goal" is a real quantity
        self._yaw += w * dt
        step_dist = v * dt * float(np.cos(np.clip(self._yaw, -np.pi, np.pi)))
        self._dist += step_dist

        r = (self.w.progress * step_dist
             + self.w.alive
             - self.w.risk * max(0.0, self._risk - self.w.risk_deadband)
             - self.w.unknown * max(0.0, self.w.trav_ok - self._trav)
             - self.w.jerk * float(a != self._last_a)
             - self.w.idle * float(v < 0.10)
             - self.w.heading * abs(self._yaw))
        self._last_a = a
        self._t += 1

        info = {"risk": self._risk, "trav": self._trav, "progress": self._dist,
                "action": a}

        if self.track_map:
            # Geometric ground truth, kept completely out of the reward: the map
            # is warped kinematically alongside the latent so we can score the
            # policy against real obstacle contact rather than the world model's
            # own opinion of itself.  `geometric_collision` only counts cells the
            # footprint newly *enters* while moving, so a stationary policy can
            # never register a hit (it previously did, on every episode, which
            # made the whole comparison uninformative).
            self._true_hits += int(bu.geometric_collision(self._map, a))
            self._map = bu.warp_state(self._map, v, w, dt)
            self._true_risk = bu.collision_risk_target(self._map, a)
            info["true_risk"] = self._true_risk
            info["true_hits"] = self._true_hits

        terminated = self._risk >= CFG.safety.risk_stop
        if terminated:
            r -= self.w.collision
            info["collision"] = True
        truncated = self._t >= self.max_steps
        return self._obs(), float(r), bool(terminated), bool(truncated), info


# ---------------------------------------------------------------- init bank

def state_context(states: np.ndarray) -> np.ndarray:
    """(N,8,H,W) -> (N,3) scene context: mean confidence, unknown fraction, traversability.

    Cheap, model-free, and identical whether it is computed for PPO training or
    for a live frame, so the observation layout cannot drift between the two.
    """
    if states.ndim == 3:
        states = states[None]
    ctx = []
    for s in states:
        obs = s[bu.CH_OBS] > 0.5
        conf = float(s[bu.CH_CONF][obs].mean()) if obs.any() else 0.0
        unk = float(np.mean((s[bu.CH_UNK] > 0.5) | (s[bu.CH_CONF] < CFG.safety.conf_unknown)))
        ctx.append([conf, unk, bu.mean_traversability(s)])
    return np.asarray(ctx, np.float32)


@torch.no_grad()
def encode_bank(wm: DrishtiWorldModel, states: np.ndarray, device: str | None = None,
                batch: int = 16):
    """(N,8,H,W) world states -> (latents (N,96), ctx (N,3)).

    `device=None` keeps the model where it already is (important: this is called
    with a CUDA-resident stage model during inference).
    """
    dev = next(wm.parameters()).device if device is None else torch.device(device)
    if next(wm.parameters()).device != dev:
        wm = wm.to(dev)
    wm.eval()
    lat = []
    for i in range(0, states.shape[0], batch):
        x = torch.from_numpy(states[i:i + batch].astype(np.float32)).to(dev)
        lat.append(wm.encode(x).float().cpu().numpy())
    return np.concatenate(lat, 0).astype(np.float32), state_context(states)


def collect_init_states(max_per_clip: int = 60, n_synth: int = 240,
                        seed: int = CFG.seed) -> tuple[np.ndarray, str]:
    """Real cached BEV world states if available, synthetic stand-ins otherwise."""
    from ..config import CLIP_IDS
    real = []
    used = []
    for cid in CLIP_IDS:
        s = bu.load_bev_states(cid)
        if s is None:
            continue
        step = max(1, s.shape[0] // max_per_clip)
        real.append(s[::step][:max_per_clip])
        used.append(cid)
    if real:
        arr = np.concatenate(real, 0).astype(np.float32)
        return arr, "real cached BEV maps: " + ", ".join(used)
    rng = np.random.default_rng(seed)
    seqs = [bu.synthetic_sequence(20, "mixed", seed=int(rng.integers(1e6)))
            for _ in range(max(1, n_synth // 20))]
    return np.concatenate(seqs, 0).astype(np.float32), \
        "synthetic stand-in scenes (no `bev` cache present yet)"


def make_env(wm: DrishtiWorldModel, states: np.ndarray, device: str = "cpu",
             track_map: bool = False, seed: int = CFG.seed, **kw) -> WorldModelEnv:
    lat, ctx = encode_bank(wm, states, device)
    return WorldModelEnv(wm, lat, ctx, states if track_map else None,
                         device=device, track_map=track_map, seed=seed, **kw)


# ---------------------------------------------------------------- policy stage

class PolicyStage:
    """Loads the PPO policy and proposes one action per frame.

    The supervisor gates whatever this returns - `PolicyStage` never commands the
    vehicle directly.  If no checkpoint exists it falls back to a documented
    heuristic (prefer FORWARD, slow when traversability is low) and says so via
    `self.source`.
    """

    name = "rl_policy"

    def __init__(self, device: str = "cpu", ckpt: Optional[Path] = None,
                 wm: Optional[DrishtiWorldModel] = None):
        self.device = torch.device("cpu" if device == "cpu" else device)
        self.model = None
        self.source = "heuristic fallback (no PPO checkpoint)"
        p = Path(ckpt) if ckpt is not None else PPO_CKPT
        if p.exists():
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(str(p), device="cpu")
                self.source = str(p)
            except Exception as e:            # pragma: no cover - reported, not hidden
                self.source = f"PPO load failed ({type(e).__name__}: {e})"
        self.wm = wm
        self._last_a = ACTIONS.index("STOP")

    def reset(self) -> None:
        self._last_a = ACTIONS.index("STOP")

    def build_obs(self, latent: np.ndarray, ctx: np.ndarray, trav: float,
                  risk: float, yaw_err: float = 0.0, t_frac: float = 0.0) -> np.ndarray:
        last = np.zeros(N_ACTIONS, np.float32)
        last[self._last_a] = 1.0
        return np.concatenate([
            np.asarray(latent, np.float32).ravel(),
            np.asarray(ctx, np.float32).ravel()[:OBS_CTX],
            np.array([trav, risk, np.clip(yaw_err / np.pi, -1, 1)], np.float32),
            last, np.array([t_frac], np.float32)]).astype(np.float32)

    def act(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        """-> (action, probabilities over the 6 actions)."""
        if self.model is None:
            probs = np.full(N_ACTIONS, 0.02, np.float32)
            trav = float(obs[OBS_LATENT + OBS_CTX])
            risk = float(obs[OBS_LATENT + OBS_CTX + 1])
            probs[ACTIONS.index("FORWARD")] = max(0.05, trav - risk)
            probs[ACTIONS.index("SLOW")] = max(0.05, 0.6 - trav + risk)
            probs[ACTIONS.index("STOP")] = max(0.02, risk - 0.4)
            probs /= probs.sum()
            a = int(np.argmax(probs))
        else:
            with torch.no_grad():
                ot, _ = self.model.policy.obs_to_tensor(obs.astype(np.float32))
                dist = self.model.policy.get_distribution(ot)
                probs = dist.distribution.probs.cpu().numpy().ravel().astype(np.float32)
            a = int(np.argmax(probs))
        self._last_a = a
        return a, probs

    def __call__(self, packet: FramePacket) -> FramePacket:
        """Needs `packet.bev`; stores the proposal on the packet for the supervisor."""
        if packet.bev is None or self.wm is None:
            return packet
        st = bu.pack_state(packet.bev.height, packet.bev.trav_prob, packet.bev.conf,
                           packet.bev.age, getattr(packet.bev, "hits", None))
        lat, ctx = encode_bank(self.wm, st[None], str(self.device))
        trav = bu.mean_traversability(st)
        risk = float(max((p.risk_total for p in packet.wm_preds), default=0.0))
        a, probs = self.act(self.build_obs(lat[0], ctx[0], trav, risk))
        packet._policy_action = a          # consumed by Supervisor.__call__
        packet._policy_probs = probs
        return packet


if __name__ == "__main__":
    import time
    states, src = collect_init_states(n_synth=80)
    print(f"[init states] {states.shape[0]} scenes from {src}")
    wm = DrishtiWorldModel()
    if WM_CKPT.exists():
        sd = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
        wm.load_state_dict(sd["model"] if "model" in sd else sd)
        print(f"[world model] loaded {WM_CKPT}")
    env = make_env(wm, states, "cpu", track_map=True, max_steps=32)
    print(f"[env] obs {env.observation_space.shape} act {env.action_space.n} "
          f"dt {bu.WM_DT:.3f} s, episode {env.max_steps} steps "
          f"= {env.max_steps*bu.WM_DT:.1f} s")
    for name, pol in [("always FORWARD", lambda o: 0), ("always STOP", lambda o: 5),
                      ("random", lambda o: int(np.random.default_rng().integers(0, 6)))]:
        R, P, C, n = 0.0, 0.0, 0, 12
        t0 = time.perf_counter()
        for e in range(n):
            o, _ = env.reset(options={"index": e * 7})
            done = False
            while not done:
                o, r, term, trunc, info = env.step(pol(o))
                R += r
                done = term or trunc
            P += info["progress"]
            C += int(info.get("true_hits", 0) > 0)
        dt = (time.perf_counter() - t0) / n * 1e3
        print(f"  {name:15s} return {R/n:+7.2f}  progress {P/n:5.2f} m  "
              f"geometric-collision episodes {C}/{n}  ({dt:.1f} ms/episode)")
