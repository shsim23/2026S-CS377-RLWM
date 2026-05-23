import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def variance_regularization(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    z_flat = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z_flat.var(dim=0) + 1e-4)
    return F.relu(target_std - std).mean()


def weight_init(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity="linear")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.GRUCell):
        for name, param in m.named_parameters():
            if "weight" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
