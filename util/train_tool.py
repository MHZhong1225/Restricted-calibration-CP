import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import conformal_quantile, kl_diagonal_gaussians


def train_backbone(model, train_loader, epochs=150, lr=1e-3, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, total_n = 0.0, 0

        for x, y, *_attrs in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        if (epoch + 1) % 25 == 0:
            print(f"[Backbone] Epoch {epoch + 1:03d} | loss={total_loss / total_n:.4f}")

    return model



def train_stochastic_assignment(
    model,
    train_loader,
    backbone,
    alpha=0.1,
    epochs=100,
    lr=1e-3,
    beta_kl=1e-3,
    lambda_balance=1e-2,
    lambda_score=0.5,
    lambda_tail=1.0,
    lambda_miss=1.0,
    lambda_difficulty=0.5,
    lambda_proto_risk=0.1,
    tail_quantile=0.9,
    n_latent_samples=8,
    device="cpu",
):
    model = model.to(device)
    backbone = backbone.to(device)
    backbone.eval()

    for p in model.backbone.parameters():
        p.requires_grad = False

    all_scores = []
    with torch.no_grad():
        for x, y, *_attrs in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = backbone(x)
            probs = F.softmax(logits, dim=-1)
            s = 1.0 - probs[torch.arange(len(y), device=device), y]
            all_scores.append(s.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    tail_threshold = float(np.quantile(all_scores, tail_quantile))
    global_cp_threshold = conformal_quantile(all_scores, alpha)

    tail_threshold_t = torch.tensor(tail_threshold, dtype=torch.float32, device=device)
    global_cp_threshold_t = torch.tensor(global_cp_threshold, dtype=torch.float32, device=device)

    print(f"[SLS] tail_threshold={tail_threshold:.4f} | global_cp_threshold={global_cp_threshold:.4f}")
    num_prototypes = model.prototypes.shape[0]
    score_head = nn.Sequential(nn.Linear(num_prototypes, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    tail_head = nn.Sequential(nn.Linear(num_prototypes, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    miss_head = nn.Sequential(nn.Linear(num_prototypes, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(
        trainable_params + list(score_head.parameters()) + list(tail_head.parameters()) + list(miss_head.parameters()),
        lr=lr,
    )

    for epoch in range(epochs):
        model.train()
        total_loss, total_n = 0.0, 0

        for x, y, *_attrs in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                logits, _ = backbone(x)
                probs = F.softmax(logits, dim=-1)
                target_score = 1.0 - probs[torch.arange(len(y), device=device), y]
                target_tail = (target_score >= tail_threshold_t).float()
                target_miss = (target_score > global_cp_threshold_t).float()

            out = model(x, n_latent_samples=n_latent_samples)
            post_mu, post_sig = out["post_mu"], out["post_sig"]
            prior_mu, prior_sig = out["prior_mu"], out["prior_sig"]
            avg_weights = out["avg_weights"]
            difficulty = out["difficulty"]

            pred_score = score_head(avg_weights).squeeze(-1)
            pred_tail_logit = tail_head(avg_weights).squeeze(-1)
            pred_miss_logit = miss_head(avg_weights).squeeze(-1)

            score_loss = F.mse_loss(pred_score, target_score)
            tail_loss = F.binary_cross_entropy_with_logits(pred_tail_logit, target_tail)
            miss_loss = F.binary_cross_entropy_with_logits(pred_miss_logit, target_miss)
            difficulty_loss = F.mse_loss(pred_score.sigmoid(), target_score.detach()) + F.mse_loss(
                difficulty, target_score.detach()
            )
            kl_loss = kl_diagonal_gaussians(post_mu, post_sig, prior_mu, prior_sig).mean()

            proto_mass = avg_weights.mean(dim=0)
            uniform = torch.full_like(proto_mass, 1.0 / proto_mass.numel())
            balance_loss = F.kl_div(
                (proto_mass + 1e-8).log(),
                uniform,
                reduction="batchmean",
                log_target=False,
            )

            proto_tail_mass = (avg_weights * target_tail.unsqueeze(1)).sum(dim=0)
            proto_assign_mass = avg_weights.sum(dim=0) + 1e-8
            proto_tail_rate = proto_tail_mass / proto_assign_mass
            proto_risk_sep_loss = -proto_tail_rate.var()

            loss = (
                lambda_score * score_loss
                + lambda_tail * tail_loss
                + lambda_miss * miss_loss
                + lambda_difficulty * difficulty_loss
                + lambda_proto_risk * proto_risk_sep_loss
                + beta_kl * kl_loss
                + lambda_balance * balance_loss
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        if (epoch + 1) % 20 == 0:
            print(f"[SLS] Epoch {epoch + 1:03d} | loss={total_loss / total_n:.4f}")

    return model



def train_soft_prototype_assignment(model, train_loader, epochs=80, lr=1e-2, lambda_balance=1.0, device="cpu"):
    model = model.to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam([model.prototypes], lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, total_n = 0.0, 0

        for x, _y, *_attrs in train_loader:
            x = x.to(device)
            out = model(x)
            weights = out["weights"]
            proto_mass = weights.mean(dim=0)
            uniform = torch.full_like(proto_mass, 1.0 / proto_mass.numel())
            loss = lambda_balance * F.kl_div(
                (proto_mass + 1e-8).log(),
                uniform,
                reduction="batchmean",
                log_target=False,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        if (epoch + 1) % 20 == 0:
            print(f"[SoftProto] Epoch {epoch + 1:03d} | loss={total_loss / total_n:.4f}")

    return model
