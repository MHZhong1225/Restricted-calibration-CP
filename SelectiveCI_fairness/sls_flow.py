import torch
import torch.nn as nn
import torch.nn.functional as F



class Backbone(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, feature_dim=32, num_classes=6):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        h = self.feature_extractor(x)
        logits = self.classifier(h)
        return logits, h


# =========================
# 3. Utility networks
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden=2):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(num_hidden):
            layers += [nn.Linear(last, hidden_dim), nn.ReLU()]
            last = hidden_dim
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StochasticMLP(nn.Module):
    """
    sigma = softplus(raw_sigma) + min_sig
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden=2, min_sig=1e-3):
        super().__init__()
        self.network = MLP(in_dim, hidden_dim, 2 * out_dim, num_hidden)
        self.out_dim = out_dim
        self.min_sig = min_sig

    def forward(self, x):
        out = self.network(x)
        mu = out[:, :self.out_dim]
        sigma = F.softplus(out[:, self.out_dim:]) + self.min_sig
        return mu, sigma

class SoftPrototypeAssignment(nn.Module):
    def __init__(self, backbone: Backbone, feature_dim=32, num_prototypes=8, temperature=1.0):
        super().__init__()
        self.backbone = backbone
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.temperature = temperature

    def forward(self, x):
        with torch.no_grad():
            _, h = self.backbone(x)
        dist2 = ((h.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2).sum(dim=-1)
        weights = F.softmax(-dist2 / self.temperature, dim=-1)
        return {"features": h, "weights": weights}


class StochasticAssignment(nn.Module):
    """
    posterior: q(z|h)
    prior:     p(z|backbone_probs)
    routing:   prototype assignment on z
    difficulty: train-only difficulty score d(x) used for binning
    """

    def __init__(
        self,
        backbone: Backbone,
        feature_dim=32,
        latent_dim=8,
        num_prototypes=8,
        temperature=1.0,
        stochastic_hidden_dim=64,
        stochastic_num_hidden=2,
        prior_in_dim=6,
        min_sig=1e-3,
    ):
        super().__init__()
        self.backbone = backbone

        self.posterior_net = StochasticMLP(
            in_dim=feature_dim,
            hidden_dim=stochastic_hidden_dim,
            out_dim=latent_dim,
            num_hidden=stochastic_num_hidden,
            min_sig=min_sig,
        )
        self.prior_net = StochasticMLP(
            in_dim=prior_in_dim,
            hidden_dim=stochastic_hidden_dim,
            out_dim=latent_dim,
            num_hidden=stochastic_num_hidden,
            min_sig=min_sig,
        )

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))
        self.temperature = temperature
        self.difficulty_head = nn.Sequential(
            nn.Linear(num_prototypes, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def reparameterize(self, mu, sigma, n_samples=5):
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        return mu.unsqueeze(0) + eps * sigma.unsqueeze(0)

    def soft_assign(self, z):
        dist2 = ((z.unsqueeze(2) - self.prototypes.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)
        return F.softmax(-dist2 / self.temperature, dim=-1)

    def forward(self, x, n_latent_samples=8):
        with torch.no_grad():
            logits, h = self.backbone(x)
            probs = F.softmax(logits, dim=-1)

        post_mu, post_sig = self.posterior_net(h)
        prior_mu, prior_sig = self.prior_net(probs.detach())
        z = self.reparameterize(post_mu, post_sig, n_samples=n_latent_samples)
        weights = self.soft_assign(z)
        avg_weights = weights.mean(dim=0)
        difficulty = torch.sigmoid(self.difficulty_head(avg_weights)).squeeze(-1)

        return {
            "features": h,
            "backbone_probs": probs,
            "post_mu": post_mu,
            "post_sig": post_sig,
            "prior_mu": prior_mu,
            "prior_sig": prior_sig,
            "z": z,
            "weights": weights,
            "avg_weights": avg_weights,
            "difficulty": difficulty,
        }

