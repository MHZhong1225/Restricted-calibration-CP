import numpy as np
import torch
import torch.nn.functional as F
from util.utils import kl_diagonal_gaussians


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
    epochs=100,
    lr=1e-3,
    beta_kl=1e-3,
    lambda_balance=1e-2,
    lambda_score=0.5,
    n_latent_samples=8,
    num_score_bins=20,
    score_bin_edges="quantile",
    hist_smoothing=1e-3,
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
    num_prototypes = model.prototypes.shape[0]
    num_score_bins = int(num_score_bins)
    if num_score_bins < 2:
        raise ValueError(f"num_score_bins must be >=2, got {num_score_bins}")

    if isinstance(score_bin_edges, str) and score_bin_edges == "quantile":
        edges = np.quantile(all_scores, np.linspace(0.0, 1.0, num_score_bins + 1))
        edges[0] = 0.0
        edges[-1] = 1.0
        for i in range(1, len(edges)):
            edges[i] = max(edges[i], edges[i - 1])
        edges = edges.astype(np.float32)
    elif isinstance(score_bin_edges, str) and score_bin_edges == "uniform":
        edges = np.linspace(0.0, 1.0, num_score_bins + 1, dtype=np.float32)
    else:
        edges = np.asarray(score_bin_edges, dtype=np.float32)
        if edges.ndim != 1 or len(edges) != num_score_bins + 1:
            raise ValueError("score_bin_edges must be 'quantile'|'uniform' or array of length num_score_bins+1")

    edges_mid_t = torch.tensor(edges[1:-1], dtype=torch.float32, device=device)
    hist_smoothing = float(hist_smoothing)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=lr)

    for epoch in range(epochs):
        model.eval()
        with torch.no_grad():
            counts = torch.full(
                (num_prototypes, num_score_bins),
                float(hist_smoothing),
                dtype=torch.float32,
                device=device,
            )
            for x, y, *_attrs in train_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = backbone(x)
                probs = F.softmax(logits, dim=-1)
                target_score = 1.0 - probs[torch.arange(len(y), device=device), y]
                score_bin = torch.bucketize(target_score, edges_mid_t, right=False)
                out = model(x, n_latent_samples=n_latent_samples)
                avg_weights = out["avg_weights"]

                bin_oh = F.one_hot(score_bin, num_classes=num_score_bins).to(dtype=torch.float32)
                counts += avg_weights.transpose(0, 1) @ bin_oh

            p_gb = counts / counts.sum(dim=1, keepdim=True).clamp(min=1e-12)

        model.train()
        total_loss, total_n = 0.0, 0

        for x, y, *_attrs in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                logits, _ = backbone(x)
                probs = F.softmax(logits, dim=-1)
                target_score = 1.0 - probs[torch.arange(len(y), device=device), y]
                score_bin = torch.bucketize(target_score.detach(), edges_mid_t, right=False)

            out = model(x, n_latent_samples=n_latent_samples)
            post_mu, post_sig = out["post_mu"], out["post_sig"]
            prior_mu, prior_sig = out["prior_mu"], out["prior_sig"]
            avg_weights = out["avg_weights"]

            mix_bin_probs = torch.sum(avg_weights * p_gb[:, score_bin].transpose(0, 1), dim=1).clamp(min=1e-12)
            score_nll = -torch.log(mix_bin_probs).mean()
            kl_loss = kl_diagonal_gaussians(post_mu, post_sig, prior_mu, prior_sig).mean()

            proto_mass = avg_weights.mean(dim=0)
            uniform = torch.full_like(proto_mass, 1.0 / proto_mass.numel())
            balance_loss = F.kl_div(
                (proto_mass + 1e-8).log(),
                uniform,
                reduction="batchmean",
                log_target=False,
            )

            loss = (
                lambda_score * score_nll
                + beta_kl * kl_loss
                + lambda_balance * balance_loss
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        if (epoch + 1) % 20 == 0:
            print(f"[SGCP] Epoch {epoch + 1:03d} | loss={total_loss / total_n:.4f}")

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
