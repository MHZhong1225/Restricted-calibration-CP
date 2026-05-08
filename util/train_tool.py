import math

import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split


def split_train_loader(train_loader, seed=42):
    """Splits a DataLoader's dataset into two loaders for model training and transformation."""
    dataset = train_loader.dataset
    bs = train_loader.batch_size

    if len(dataset) < 2:
        raise ValueError(f"Required at least 2 samples, got {len(dataset)}.")

    len1 = len(dataset) // 2
    len2 = len(dataset) - len1
    ds1, ds2 = random_split(dataset, [len1, len2], generator=torch.Generator().manual_seed(seed))

    loader_train_model = DataLoader(ds1, batch_size=bs, shuffle=True, drop_last=False)
    loader_transform = DataLoader(ds2, batch_size=bs, shuffle=False, drop_last=False)
    return loader_train_model, loader_transform


def weighted_cvm_uniform_loss(u: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculates weighted Cramer-von Mises loss for Uniform(0,1) targets."""
    w = w / w.sum().clamp_min(eps)
    u_sorted, idx = torch.sort(u)
    w_sorted = w[idx]

    cdf_emp = torch.cumsum(w_sorted, dim=0)
    cdf_tgt = (2 * torch.arange(1, len(u_sorted) + 1, device=u.device) - 1) / (2 * len(u_sorted))
    return torch.sum(w_sorted * (cdf_emp - cdf_tgt) ** 2)


def weighted_rank_uniform_loss(values, weights, eps=1e-8, min_mass=1e-6):
    """Encourages weighted empirical distribution to match Uniform(0,1) via rank midpoints."""
    values = values.clamp(eps, 1.0 - eps)
    weights = weights.clamp_min(0.0)
    mass = weights.sum()

    if mass.detach().item() <= min_mass or values.numel() <= 1:
        return values.new_zeros(())

    order = torch.argsort(values)
    v_sorted, w_sorted = values[order], weights[order]

    pi = w_sorted / (mass + eps)
    pi_target = pi.detach()
    rank_midpoint = torch.cumsum(pi_target, dim=0) - 0.5 * pi_target

    return torch.sum(pi * (v_sorted - rank_midpoint) ** 2)


def grouping_mi_loss(weights, eta=1.0, eps=1e-12):
    """Computes grouping loss via sample-level entropy minimization and batch-level maximization."""
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)

    sample_entropy = -torch.sum(weights * torch.log(weights.clamp_min(eps)), dim=1).mean()
    mean_w = weights.mean(dim=0)
    batch_entropy = -torch.sum(mean_w * torch.log(mean_w.clamp_min(eps)))

    return sample_entropy - eta * batch_entropy, sample_entropy, batch_entropy


def _cache_backbone_features(model, loader, device):
    """Helper to pre-compute backbone features to accelerate stochastic assignment training."""
    model.backbone.eval()
    h_list, p_list, y_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError(
                    f"Unexpected batch format for feature caching: "
                    f"type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'n/a'}"
                )
            logits, h = model.backbone(x.to(device))
            h_list.append(h.cpu())
            p_list.append(F.softmax(logits, dim=-1).cpu())
            y_list.append(y.cpu())

    return TensorDataset(torch.cat(h_list), torch.cat(p_list), torch.cat(y_list))



def train_stochastic_assignment(
    model,
    train_loader,
    epochs=100,
    lr=1e-3,
    lambda_rank=1.0,
    lambda_nll=0.2,
    lambda_balance=1e-2,
    n_latent_samples=8,
    device="cpu",
):
    model.to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False

    feat_ds = _cache_backbone_features(model, train_loader, device)
    feat_loader = DataLoader(feat_ds, batch_size=train_loader.batch_size or 128, shuffle=True)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    early_stop_metric = "rank"
    early_stop_patience = 25
    early_stop_min_delta = 1e-4
    early_stop_min_epochs = 30
    early_stop_max_w = 2
    best_metric, patience_counter = float("inf"), 0
    num_prototypes = model.prototypes.shape[0]
    mu_k, tau_k = model.mu_k, model.tau_k
    eps, log_k = 1e-8, math.log(num_prototypes)

    for epoch in range(epochs):
        model.train()
        metrics = {k: 0.0 for k in ["loss", "rank", "nll", "bal", "ent", "group", "maxp", "n"]}

        for h_batch, probs_batch, y_batch in feat_loader:
            h_batch, probs_batch, y_batch = h_batch.to(device), probs_batch.to(device), y_batch.to(device)
            bs = len(y_batch)

            out = model(x=None, h=h_batch, probs=probs_batch, n_latent_samples=n_latent_samples)
            weights = out["avg_weights"]
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)

            batch_idx = torch.arange(bs, device=device)
            target_score = 1.0 - probs_batch[batch_idx, y_batch]

            mu_k_bounded, tau_k_pos = torch.sigmoid(mu_k), F.softplus(tau_k) + 1e-3
            G_k = torch.sigmoid((target_score.unsqueeze(1) - mu_k_bounded) / tau_k_pos)

            loss_rank, active_c = torch.zeros((), device=device), 0
            for k in range(num_prototypes):
                if weights[:, k].sum().item() > 1e-6:
                    loss_rank += weighted_rank_uniform_loss(G_k[:, k], weights[:, k])
                    active_c += 1
            loss_rank = loss_rank / max(active_c, 1)

            g_k = G_k * (1.0 - G_k) / tau_k_pos
            g_i = torch.sum(weights * g_k, dim=1).clamp_min(eps)
            loss_nll = -torch.log(g_i).mean()

            proto_mass = weights.mean(dim=0).clamp_min(eps)
            proto_mass /= proto_mass.sum()
            loss_balance = torch.sum(proto_mass * (torch.log(proto_mass) + math.log(float(num_prototypes))))
            loss_balance = loss_balance / log_k

            sample_entropy = -torch.sum(weights * weights.clamp_min(eps).log(), dim=1).mean()
            loss_entropy = sample_entropy / max(log_k, eps)

            entropy_gate = (1.0 - loss_balance.detach()).clamp(0, 1) if epoch >= 10 else 0.0
            loss_group = loss_balance + 0.03 * entropy_gate * loss_entropy

            loss = lambda_rank * loss_rank + lambda_nll * loss_nll + lambda_balance * loss_group

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            metrics["loss"] += loss.item() * bs
            metrics["rank"] += loss_rank.item() * bs
            metrics["nll"] += loss_nll.item() * bs
            metrics["bal"] += loss_balance.item() * bs
            metrics["ent"] += sample_entropy.item() * bs
            metrics["group"] += loss_group.item() * bs
            metrics["maxp"] += weights.max(dim=1).values.mean().item() * bs
            metrics["n"] += bs

        avg = {k: v / metrics["n"] for k, v in metrics.items() if k != "n"}

        if (epoch + 1) % 10 == 0:
            print(
                f"[SGCP] Ep {epoch+1:03d} | Total:{avg['loss']:.4f} | "
                f"Rank:{avg['rank']:.4f} | NLL:{avg['nll']:.4f} | MaxW:{avg['maxp']:.4f}"
            )

        monitor_value = avg.get(early_stop_metric)
        if monitor_value is None:
            raise ValueError(
                f"Unknown early_stop_metric={early_stop_metric!r}. "
                f"Available metrics: {sorted(avg.keys())}"
            )

        if epoch + 1 >= int(early_stop_min_epochs):
            if avg["maxp"] >= float(early_stop_max_w):
                print(
                    f"[SGCP] Early stop at epoch {epoch+1}: "
                    f"MaxW={avg['maxp']:.4f} >= {float(early_stop_max_w):.4f}"
                )
                break
            if monitor_value < best_metric - early_stop_min_delta:
                best_metric, patience_counter = monitor_value, 0
            elif (patience_counter := patience_counter + 1) >= early_stop_patience:
                print(
                    f"[SGCP] Early stop at epoch {epoch+1}: "
                    f"{early_stop_metric} did not improve for {early_stop_patience} epochs "
                    f"(best={best_metric:.4f}, current={monitor_value:.4f})"
                )
                break
        elif monitor_value < best_metric:
            best_metric = monitor_value

    return model


def train_backbone(model, train_loader, epochs=150, lr=1e-3, device="cpu"):
    """Standard supervised training for the backbone model."""
    model.to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total_n = 0.0, 0, 0
        for batch in train_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError(
                    f"Unexpected batch format for backbone training: "
                    f"type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'n/a'}"
                )
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total_n += x.size(0)

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            acc = correct / total_n
            print(f"[Backbone] Ep {epoch+1:03d} | Loss:{total_loss/total_n:.4f} | Acc:{acc:.4f}")
    return model


def train_prototype_assignment(model, train_loader, epochs=80, lr=1e-2, lambda_balance=1.0, device="cpu"):
    """Optimizes prototype positions by enforcing balanced component assignment."""
    model.to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = Adam([model.prototypes], lr=lr)
    uniform_tgt = torch.full((model.prototypes.shape[0],), 1.0 / model.prototypes.shape[0], device=device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            out = model(batch[0].to(device))
            proto_mass = out["weights"].mean(dim=0)
            loss = lambda_balance * F.kl_div((proto_mass + 1e-8).log(), uniform_tgt, reduction="batchmean")

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"[Proto] Ep {epoch+1:03d} | Loss:{total_loss/len(train_loader):.4f}")
    return model
