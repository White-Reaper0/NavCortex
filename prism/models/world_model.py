"""prism tiny neural world model - latent dynamics over the BEV world state.

Why a latent world model instead of generated future video
----------------------------------------------------------
The planner does not need photorealistic pixels of the future.  It needs to know,
for each candidate action, *where the obstacles will be, how drivable the ground
ahead is, and how likely a collision is*.  Generating future RGB frames to answer
that would cost orders of magnitude more compute than the whole rest of the stack
and would still have to be re-parsed back into geometry.  prism instead
compresses the 8-channel 128x192 BEV world state into a 96-D latent and learns a
GRU that advances that latent under a candidate action.  The resulting model is
small enough (see the parameter count printed by the self-test) that rolling out
all six actions for six steps costs well under a millisecond on GPU, which is
what makes it usable as the *simulator* for RL and as a per-frame risk oracle in
the safety supervisor.

Shapes
------
    encoder   (B, 8, 128, 192) -> (B, 96)
    dynamics  (B, 96) x action -> (B, 96)              GRU, hidden 192, 2 layers
    heads     (B, 96) -> occupancy (B, 16, 16)
                       -> mean traversability (B,)     sigmoid, [0,1]
                       -> collision risk (B,)          sigmoid, [0,1]

All dimensions come from `CFG.wm`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, CKPT_DIR, ACTIONS, N_ACTIONS
from ..types import FramePacket, WorldModelPrediction
from ..nav import bev_utils as bu

WM_CKPT = CKPT_DIR / "world_model.pt"


# ------------------------------------------------------------------ encoder

def _conv(cin, cout, stride=2):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


class BEVEncoder(nn.Module):
    """Small strided conv trunk: (8, 128, 192) -> (state_dim,).

    128x192 -> 64x96 -> 32x48 -> 16x24 -> 8x12, adaptively pooled to 3x4 so the
    head stays tiny.  ~0.18 M parameters.
    """

    def __init__(self, in_ch: int = bu.BEV_CH, state_dim: int = CFG.wm.state_dim):
        super().__init__()
        self.stem = nn.Sequential(
            _conv(in_ch, 16, 2),      # 64 x 96
            _conv(16, 32, 2),         # 32 x 48
            _conv(32, 64, 2),         # 16 x 24
            _conv(64, 96, 2),         # 8 x 12
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 4, state_dim),
            nn.LayerNorm(state_dim),
            nn.Tanh(),                # bounded latent keeps multi-step rollouts stable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.stem(x)))


# ------------------------------------------------------------------ dynamics

class LatentDynamics(nn.Module):
    """Stacked GRU cells over (state, action one-hot) predicting a residual step.

    s_{t+1} = tanh( s_t + W h_t ), h from a 2-layer GRU with hidden `CFG.wm.hidden`.
    Predicting a *residual* keeps the identity ("nothing changes") easy to express,
    which matters because at 0.33 s per step most of the map really does persist -
    the model only has to learn the flow induced by the commanded motion.
    """

    def __init__(self, state_dim: int = CFG.wm.state_dim, action_dim: int = N_ACTIONS,
                 hidden: int = CFG.wm.hidden, n_layers: int = CFG.wm.n_layers):
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.hidden, self.n_layers = hidden, n_layers
        self.inp = nn.Linear(state_dim + action_dim, hidden)
        self.cells = nn.ModuleList(
            [nn.GRUCell(hidden, hidden) for _ in range(n_layers)])
        self.out = nn.Linear(hidden, state_dim)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.out.weight, std=1e-3)     # start close to identity

    def init_hidden(self, b: int, device, dtype=torch.float32):
        return [torch.zeros(b, self.hidden, device=device, dtype=dtype)
                for _ in range(self.n_layers)]

    def step(self, s: torch.Tensor, a_onehot: torch.Tensor, hs: list):
        x = torch.tanh(self.inp(torch.cat([s, a_onehot], -1)))
        new_hs = []
        for i, cell in enumerate(self.cells):
            h = cell(x, hs[i])
            new_hs.append(h)
            x = h
        return torch.tanh(s + self.out(x)), new_hs

    def rollout(self, s0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """s0 (B, S), actions (B, T) long -> (B, T, S) predicted next states."""
        b, T = actions.shape
        hs = self.init_hidden(b, s0.device, s0.dtype)
        s = s0
        outs = []
        for t in range(T):
            a = F.one_hot(actions[:, t], self.action_dim).to(s0.dtype)
            s, hs = self.step(s, a, hs)
            outs.append(s)
        return torch.stack(outs, 1)


# ------------------------------------------------------------------ decoders

class Heads(nn.Module):
    """Latent -> 16x16 obstacle occupancy, mean traversability, collision risk."""

    def __init__(self, state_dim: int = CFG.wm.state_dim, g: int = bu.OCC_G):
        super().__init__()
        self.g = g
        self.occ = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(inplace=True),
            nn.Linear(128, g * g))
        self.trav = nn.Sequential(
            nn.Linear(state_dim, 48), nn.SiLU(inplace=True), nn.Linear(48, 1))
        self.risk = nn.Sequential(
            nn.Linear(state_dim, 48), nn.SiLU(inplace=True), nn.Linear(48, 1))

    def forward(self, s: torch.Tensor):
        occ = self.occ(s).view(*s.shape[:-1], self.g, self.g)
        return occ, self.trav(s).squeeze(-1), self.risk(s).squeeze(-1)


# ------------------------------------------------------------------ full model

class PrismWorldModel(nn.Module):
    """Encoder + latent dynamics + decode heads."""

    def __init__(self, state_dim: int = CFG.wm.state_dim):
        super().__init__()
        self.encoder = BEVEncoder(bu.BEV_CH, state_dim)
        self.dynamics = LatentDynamics(state_dim)
        self.heads = Heads(state_dim)
        self.state_dim = state_dim

    # -------- convenience
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, s: torch.Tensor):
        """-> (occ_prob, trav in [0,1], risk in [0,1])"""
        occ, trav, risk = self.heads(s)
        return torch.sigmoid(occ), torch.sigmoid(trav), torch.sigmoid(risk)

    def decode_logits(self, s: torch.Tensor):
        return self.heads(s)

    def rollout(self, s0: torch.Tensor, action: int, horizon: int = CFG.wm.pred_horizon):
        a = torch.full((s0.shape[0], horizon), int(action), dtype=torch.long,
                       device=s0.device)
        return self.dynamics.rollout(s0, a)

    def rollout_all_actions(self, s0: torch.Tensor,
                            horizon: int = CFG.wm.pred_horizon):
        """s0 (1, S) -> states (A, T, S), occ (A, T, g, g), trav (A, T), risk (A, T)."""
        a = torch.arange(N_ACTIONS, device=s0.device, dtype=torch.long)
        s_rep = s0.expand(N_ACTIONS, -1).contiguous()
        acts = a[:, None].expand(N_ACTIONS, horizon).contiguous()
        states = self.dynamics.rollout(s_rep, acts)                  # (A,T,S)
        occ, trav, risk = self.decode(states)
        return states, occ, trav, risk

    # -------- introspection
    def param_counts(self) -> dict:
        def n(m):
            return sum(p.numel() for p in m.parameters())
        return {"encoder": n(self.encoder), "dynamics": n(self.dynamics),
                "heads": n(self.heads), "total": n(self)}


# ------------------------------------------------------------------ stage

class WorldModelStage:
    """Contract stage: fills `packet.wm_preds` with one rollout per action.

    Usage:
        stage = WorldModelStage(device="cuda")
        packet = stage(packet)            # needs packet.bev
    """

    name = "world_model"

    def __init__(self, device: str = "cuda", ckpt: Optional[Path] = None,
                 horizon: int = CFG.wm.pred_horizon):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available())
                                   else "cpu")
        self.horizon = int(horizon)
        self.model = PrismWorldModel().to(self.device).eval()
        self.loaded_from = "random init (untrained)"
        p = Path(ckpt) if ckpt is not None else WM_CKPT
        if p.exists():
            sd = torch.load(p, map_location=self.device, weights_only=False)
            self.model.load_state_dict(sd["model"] if "model" in sd else sd)
            self.loaded_from = str(p)
        self.last_latency_ms = 0.0
        self.last_encode_ms = 0.0
        self._last_state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_state = None

    # -------- core
    @torch.no_grad()
    def predict(self, bev_state: np.ndarray) -> tuple[np.ndarray, list[WorldModelPrediction]]:
        """(8,H,W) world state -> (latent, [WorldModelPrediction per action])."""
        x = torch.from_numpy(np.ascontiguousarray(bev_state, np.float32))[None].to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        s0 = self.model.encode(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        states, occ, trav, risk = self.model.rollout_all_actions(s0, self.horizon)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.last_encode_ms = (t1 - t0) * 1e3
        self.last_latency_ms = (t2 - t0) * 1e3

        states = states.float().cpu().numpy()
        occ = occ.float().cpu().numpy()
        trav = trav.float().cpu().numpy()
        risk = risk.float().cpu().numpy()
        preds = []
        for a in range(N_ACTIONS):
            preds.append(WorldModelPrediction(
                action=a,
                next_states=states[a].astype(np.float32),
                occ_forecast=occ[a].astype(np.float32),
                trav_forecast=trav[a].astype(np.float32),
                collision_risk=risk[a].astype(np.float32),
                risk_total=float(np.max(risk[a])),
            ))
        return s0.float().cpu().numpy()[0], preds

    def __call__(self, packet: FramePacket) -> FramePacket:
        if packet.bev is None:
            return packet
        st = bu.pack_state(packet.bev.height, packet.bev.trav_prob, packet.bev.conf,
                           packet.bev.age, getattr(packet.bev, "hits", None))
        self._last_state = st
        _, preds = self.predict(st)
        packet.wm_preds = preds
        packet.timings_ms["world_model"] = self.last_latency_ms
        return packet

    # -------- reporting
    def benchmark(self, n: int = 60) -> dict:
        st = bu.synthetic_state("wall")
        for _ in range(8):
            self.predict(st)
        t0 = time.perf_counter()
        for _ in range(n):
            self.predict(st)
        dt = (time.perf_counter() - t0) / n * 1e3
        pc = self.model.param_counts()
        return {"device": str(self.device), "params": pc,
                "mb_fp32": pc["total"] * 4 / 1e6,
                "encode_ms": self.last_encode_ms,
                "full_ms": dt,
                "rollouts_per_call": N_ACTIONS,
                "horizon": self.horizon}


if __name__ == "__main__":
    m = PrismWorldModel()
    pc = m.param_counts()
    print("prism tiny world model")
    for k, v in pc.items():
        print(f"  {k:9s} {v:>9,d} params")
    print(f"  fp32 size {pc['total']*4/1e6:.2f} MB")

    x = torch.randn(2, bu.BEV_CH, CFG.bev.H, CFG.bev.W)
    s = m.encode(x)
    st = m.rollout(s, 0)
    occ, trav, risk = m.decode(st)
    print(f"  encode {tuple(x.shape)} -> {tuple(s.shape)}; rollout -> {tuple(st.shape)}")
    print(f"  occ {tuple(occ.shape)} in [{occ.min():.3f},{occ.max():.3f}]  "
          f"trav {tuple(trav.shape)}  risk {tuple(risk.shape)}")

    for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        stg = WorldModelStage(device=dev)
        b = stg.benchmark(40)
        print(f"  [{dev}] all-{N_ACTIONS}-action x {b['horizon']}-step rollout + encode: "
              f"{b['full_ms']:.3f} ms  (encode {b['encode_ms']:.3f} ms)  ckpt={stg.loaded_from}")
        _, preds = stg.predict(bu.synthetic_state("wall"))
        print("        risk_total per action: "
              + "  ".join(f"{ACTIONS[p.action]}={p.risk_total:.3f}" for p in preds))
