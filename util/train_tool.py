import numpy as np
import torch
import torch.nn.functional as F
from util.utils import kl_diagonal_gaussians
import wandb
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import wandb

def train_backbone(model, train_loader, epochs=150, lr=1e-3, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss, total_n, correct = 0.0, 0, 0

        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
        
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            acc = correct / total_n
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Backbone] Epoch {epoch + 1:03d} | loss={total_loss / total_n:.4f} | acc={acc:.4f} | lr={current_lr:.6f}")
            if wandb.run is not None: 
                wandb.log({
                    "train/backbone_loss": total_loss / total_n, 
                    "train/backbone_acc": acc,
                    "epoch": epoch + 1
                })
    
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

    print("[SGCP] Pre-extracting and caching backbone features...")
    cached_h, cached_probs, cached_y = [], [], []
    
    with torch.no_grad():
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits, h = backbone(x)
            probs = F.softmax(logits, dim=-1)
            
            cached_h.append(h.cpu())
            cached_probs.append(probs.cpu())
            cached_y.append(y.cpu())

    cached_h = torch.cat(cached_h)
    cached_probs = torch.cat(cached_probs)
    cached_y = torch.cat(cached_y)
    
    batch_size = train_loader.batch_size if train_loader.batch_size else 128
    feat_dataset = TensorDataset(cached_h, cached_probs, cached_y)
    feat_loader = DataLoader(feat_dataset, batch_size=batch_size, shuffle=True)

    target_scores = 1.0 - cached_probs[torch.arange(len(cached_y)), cached_y].numpy()
    num_prototypes = model.prototypes.shape[0]
    num_score_bins = int(num_score_bins)

    if isinstance(score_bin_edges, str) and score_bin_edges == "quantile":
        edges = np.quantile(target_scores, np.linspace(0.0, 1.0, num_score_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        for i in range(1, len(edges)):
            edges[i] = max(edges[i], edges[i - 1])
        edges = edges.astype(np.float32)
    elif isinstance(score_bin_edges, str) and score_bin_edges == "uniform":
        edges = np.linspace(0.0, 1.0, num_score_bins + 1, dtype=np.float32)

    edges_mid_t = torch.tensor(edges[1:-1], dtype=torch.float32, device=device)
    hist_smoothing = float(hist_smoothing)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=lr)

    counts = torch.rand((num_prototypes, num_score_bins), dtype=torch.float32, device=device) * 0.1 + float(hist_smoothing)
    p_gb = counts / counts.sum(dim=1, keepdim=True).clamp(min=1e-12)

    print("[SGCP] Starting fast training on cached features...")
    for epoch in range(epochs):
        model.train()
        
        total_loss, total_n = 0.0, 0
        total_nll, total_kl, total_bal = 0.0, 0.0, 0.0
        
        next_counts = torch.full((num_prototypes, num_score_bins), float(hist_smoothing), dtype=torch.float32, device=device)

        for h_batch, probs_batch, y_batch in feat_loader:
            h_batch, probs_batch, y_batch = h_batch.to(device), probs_batch.to(device), y_batch.to(device)

            out = model(x=None, h=h_batch, probs=probs_batch, n_latent_samples=n_latent_samples)

            target_score = 1.0 - probs_batch[torch.arange(len(y_batch), device=device), y_batch]
            score_bin = torch.bucketize(target_score, edges_mid_t, right=False)

            mix_bin_probs = torch.sum(out["avg_weights"] * p_gb[:, score_bin].transpose(0, 1), dim=1).clamp(min=1e-12)
            score_nll = -torch.log(mix_bin_probs).mean()

            with torch.no_grad():
                posterior = out["avg_weights"] * p_gb[:, score_bin].transpose(0, 1)
                posterior = posterior / posterior.sum(dim=1, keepdim=True).clamp(min=1e-12)
                bin_oh = F.one_hot(score_bin, num_classes=num_score_bins).to(dtype=torch.float32)
                next_counts += posterior.detach().transpose(0, 1) @ bin_oh

            kl_loss = kl_diagonal_gaussians(out["post_mu"], out["post_sig"], out["prior_mu"], out["prior_sig"]).mean()

            proto_mass = out["avg_weights"].mean(dim=0)
            uniform = torch.full_like(proto_mass, 1.0 / proto_mass.numel())
            balance_loss = F.kl_div((proto_mass + 1e-8).log(), uniform, reduction="batchmean", log_target=False)

            loss = lambda_score * score_nll + beta_kl * kl_loss + lambda_balance * balance_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            opt.step()

            bs = len(y_batch)
            total_loss += loss.item() * bs
            total_nll += score_nll.item() * bs
            total_kl += kl_loss.item() * bs
            total_bal += balance_loss.item() * bs
            total_n += bs

        new_p_gb = next_counts / next_counts.sum(dim=1, keepdim=True).clamp(min=1e-12)
        momentum = 0.9 if epoch > 0 else 0.0  
        p_gb = momentum * p_gb + (1.0 - momentum) * new_p_gb

        import wandb
        if wandb.run is not None:
            wandb.log({
                "train/sgcp_total_loss": total_loss / total_n,
                "train/sgcp_score_nll": total_nll / total_n,
                "train/sgcp_kl_loss": total_kl / total_n,
                "train/sgcp_balance_loss": total_bal / total_n,
                "epoch": epoch + 1
            })

        if (epoch + 1) % 20 == 0:
            print(f"[SGCP] Epoch {epoch + 1:03d} | loss={total_loss / total_n:.4f}")
                
    return model

def train_prototype_assignment(model, train_loader, epochs=80, lr=1e-2, lambda_balance=1.0, device="cpu"):
    model = model.to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam([model.prototypes], lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, total_n = 0.0, 0

        for batch in train_loader:
            x = batch[0].to(device)
            out = model(x)
            proto_mass = out["weights"].mean(dim=0)
            uniform = torch.full_like(proto_mass, 1.0 / proto_mass.numel())
            loss = lambda_balance * F.kl_div((proto_mass + 1e-8).log(), uniform, reduction="batchmean", log_target=False)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        if (epoch + 1) % 20 == 0:
            print(f"[Proto] Epoch {epoch + 1:03d} | loss={total_loss / total_n:.4f}")

    return model
