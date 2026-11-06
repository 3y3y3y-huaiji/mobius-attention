"""End-to-end learned-gate PoC addressing the three synthetic-benchmark critiques.

Tests whether a phase gate trained purely by Cross-Entropy from random initialization
(zero manual weight copying) across multiple seeds can:
1. Suppress superseded states on 1-update (x1 -> x2) AND 2-update (x1 -> x2 -> x3)
   chains without 2*pi wrap-around resurrection;
2. Preserve an independent entity Y in the presence of a hard distractor Y_noise
   when updates to X occur in between.

Usage:
    python -m experiments.run_learned_poc
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.phase_attention import apply_rotary

# Markers: 0=INIT, 1=UPDATE, 2=NOISE
M_INIT, M_UPDATE, M_NOISE = 0, 1, 2
# Entities: 0=ENT_X, 1=ENT_Y
ENT_X, ENT_Y = 0, 1
NUM_VALS = 8


def make_batch(batch_size: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Vectorized batch of 4-slot memory sequences with hard Y distractors and 2-step X updates.

    Slots (t = 0, 1, 2, 3):
      Scenario 0 (1-update X query, tests basic revision x1 -> x2):
        slot 0: (INIT,   ENT_Y, y_true)
        slot 1: (INIT,   ENT_X, x1)
        slot 2: (UPDATE, ENT_X, x2)
        slot 3: (NOISE,  ENT_Y, y_noise)
        query:  ENT_X -> target = x2

      Scenario 1 (2-update X query, tests 2*pi resurrection on x1 -> x2 -> x3):
        slot 0: (INIT,   ENT_Y, y_true)
        slot 1: (INIT,   ENT_X, x1)
        slot 2: (UPDATE, ENT_X, x2)
        slot 3: (UPDATE, ENT_X, x3)
        query:  ENT_X -> target = x3 (scalar pi wraps 2*pi -> 0 and resurrects x1!)

      Scenario 2 (Y query with hard distractor y_noise after X updates):
        slot 0: (INIT,   ENT_Y, y_true)
        slot 1: (INIT,   ENT_X, x1)
        slot 2: (UPDATE, ENT_X, x2)
        slot 3: (NOISE,  ENT_Y, y_noise)
        query:  ENT_Y -> target = y_true (global scalar phase flips slot 0 to pi and picks y_noise!)
    """
    scenarios = torch.arange(batch_size) % 3
    # Sample distinct values x1, x2, x3 and y_true, y_noise per row via random scores argsort
    x_vals = torch.rand(batch_size, NUM_VALS, generator=generator).argsort(dim=-1)[:, :3]
    y_vals = torch.rand(batch_size, NUM_VALS, generator=generator).argsort(dim=-1)[:, :2]
    x1, x2, x3 = x_vals[:, 0], x_vals[:, 1], x_vals[:, 2]
    y_true, y_noise = y_vals[:, 0], y_vals[:, 1]

    is_two_upd = scenarios == 1
    markers = torch.tensor([M_INIT, M_INIT, M_UPDATE, M_NOISE]).unsqueeze(0).repeat(batch_size, 1)
    markers[is_two_upd, 3] = M_UPDATE

    entities = torch.tensor([ENT_Y, ENT_X, ENT_X, ENT_Y]).unsqueeze(0).repeat(batch_size, 1)
    entities[is_two_upd, 3] = ENT_X

    values = torch.stack((y_true, x1, x2, torch.where(is_two_upd, x3, y_noise)), dim=1)
    query_entity = torch.where(scenarios == 2, ENT_Y, ENT_X)
    targets = torch.where(scenarios == 0, x2, torch.where(scenarios == 1, x3, y_true))

    return {
        "markers": markers,
        "entities": entities,
        "values": values,
        "query_entity": query_entity,
        "targets": targets,
        "scenarios": scenarios,
    }


class LearnedPhaseMemory(nn.Module):
    """Associative memory where Q/K depend only on entity identity and phase is learned."""

    def __init__(self, variant: str, d_model: int = 16):
        super().__init__()
        self.variant = variant
        # Orthogonal entity keys and value embeddings so Q-K matching is strictly by entity
        self.entity_emb = nn.Embedding(2, 4)
        self.marker_emb = nn.Embedding(3, d_model)
        self.value_emb = nn.Embedding(NUM_VALS, d_model)
        self.out_proj = nn.Linear(d_model, NUM_VALS, bias=False)

        # Learned transition detector and entity router (randomly initialized, NO weight.copy_!)
        self.update_gate = nn.Linear(d_model, 1)
        self.noise_gate = nn.Linear(d_model, 1)
        self.entity_router = nn.Linear(4, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bsz = batch["markers"].shape[0]
        q_ent = self.entity_emb(batch["query_entity"]).view(bsz, 1, 1, 4)
        k_ent = self.entity_emb(batch["entities"]).view(bsz, 1, 4, 4)
        q = F.normalize(q_ent, dim=-1) * 3.0
        k = F.normalize(k_ent, dim=-1) * 3.0
        v = self.value_emb(batch["values"]).unsqueeze(1)  # [B, 1, 4, D]

        if self.variant != "standard":
            m_feat = self.marker_emb(batch["markers"])  # [B, 4, D]
            u_step = torch.sigmoid(self.update_gate(m_feat))  # [B, 4, 1]
            n_step = torch.sigmoid(self.noise_gate(m_feat))   # [B, 4, 1]

            if self.variant == "scalar_global":
                # 1D global time-cumulative pi flips: every detected update adds pi globally
                u_hard = u_step + (u_step.round() - u_step).detach()
                cum_phi = torch.cumsum(u_hard, dim=1) * math.pi  # [B, 4, 1]
                k_angles = cum_phi.unsqueeze(1).expand(bsz, 1, 4, 4)
                q_angles = cum_phi[:, -1:, :].unsqueeze(1).expand(bsz, 1, 1, 4)
            elif self.variant == "key_harmonic":
                # Key-bound subspace routing + order-3 harmonic frequencies (2*pi/3, 4*pi/3)
                route_k = F.softmax(self.entity_router(k_ent.squeeze(1)), dim=-1)  # [B, 4, 2]
                route_q = F.softmax(self.entity_router(q_ent.squeeze(1)), dim=-1)  # [B, 1, 2]
                cum_steps = torch.cumsum(u_step * route_k, dim=1)                  # [B, 4, 2]
                k_steps = (cum_steps * route_k).sum(dim=-1, keepdim=True) + n_step * 1.5
                q_steps = (cum_steps[:, -1:, :] * route_q).sum(dim=-1, keepdim=True)

                harmonics = torch.tensor([2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0], device=q.device)
                k_planes = k_steps * harmonics.view(1, 1, 2)
                q_planes = q_steps * harmonics.view(1, 1, 2)
                k_angles = torch.cat((k_planes, k_planes), dim=-1).unsqueeze(1)
                q_angles = torch.cat((q_planes, q_planes), dim=-1).unsqueeze(1)
            else:
                raise ValueError(f"Unknown variant: {self.variant}")

            q = apply_rotary(q, q_angles.cos(), q_angles.sin())
            k = apply_rotary(k, k_angles.cos(), k_angles.sin())

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(4.0)
        weights = F.softmax(scores, dim=-1)
        readout = torch.matmul(weights, v).squeeze(1).squeeze(1)
        return self.out_proj(readout)


def train_and_evaluate(variant: str, seed: int, steps: int = 250) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    model = LearnedPhaseMemory(variant=variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-4)

    for _ in range(steps):
        batch = make_batch(120, gen)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch["targets"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    eval_gen = torch.Generator().manual_seed(seed + 9999)
    eval_batch = make_batch(900, eval_gen)
    with torch.no_grad():
        preds = model(eval_batch).argmax(dim=-1)
    correct = (preds == eval_batch["targets"]).float()
    scen = eval_batch["scenarios"]
    return {
        "seed": seed,
        "overall": round(correct.mean().item(), 4),
        "x_1update": round(correct[scen == 0].mean().item(), 4),
        "x_2updates_no_2pi": round(correct[scen == 1].mean().item(), 4),
        "y_with_distractor": round(correct[scen == 2].mean().item(), 4),
    }


def run(seeds: tuple[int, ...] = (0, 1, 2), steps: int = 250) -> dict:
    variants = ("standard", "scalar_global", "key_harmonic")
    summary = {}
    for variant in variants:
        runs = [train_and_evaluate(variant, seed, steps=steps) for seed in seeds]
        summary[variant] = {
            "seeds": runs,
            "mean_overall": round(sum(r["overall"] for r in runs) / len(runs), 4),
            "mean_x_1update": round(sum(r["x_1update"] for r in runs) / len(runs), 4),
            "mean_x_2updates_no_2pi": round(sum(r["x_2updates_no_2pi"] for r in runs) / len(runs), 4),
            "mean_y_with_distractor": round(sum(r["y_with_distractor"] for r in runs) / len(runs), 4),
        }
    return {
        "experiment": "learned_gate_poc_no_weight_copy",
        "seeds": list(seeds),
        "steps": steps,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/learned_poc_results.json")
    args = parser.parse_args()
    result = run()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Variant          Overall   X(1-upd)  X(2-upd/2pi)  Y(w/distractor)")
    for name, item in result["summary"].items():
        print(
            f"{name:15s}  {item['mean_overall']:7.4f}   {item['mean_x_1update']:8.4f}  "
            f"{item['mean_x_2updates_no_2pi']:12.4f}  {item['mean_y_with_distractor']:15.4f}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
