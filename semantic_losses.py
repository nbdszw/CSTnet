import torch
import torch.nn.functional as F

from semantic_modules import l2_normalize


def source_semantic_alignment_loss(
    zv_source: torch.Tensor,
    source_label: torch.Tensor,
    zs_prototypes: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    if metric == "cosine":
        zv = l2_normalize(zv_source)
        zs = l2_normalize(zs_prototypes)
        logits = torch.matmul(zv, zs.t())
        return F.cross_entropy(logits, source_label)
    if metric == "mse":
        target_zs = zs_prototypes[source_label]
        return F.mse_loss(zv_source, target_zs)
    raise ValueError(f"Unsupported semantic metric: {metric}")


def target_semantic_consistency_loss(
    zv_target: torch.Tensor,
    target_logits: torch.Tensor,
    zs_prototypes: torch.Tensor,
    conf_threshold: float = 0.9,
    metric: str = "cosine",
):
    probs = F.softmax(target_logits, dim=1)
    conf, pseudo = torch.max(probs, dim=1)
    mask = conf >= conf_threshold

    valid_ratio = mask.float().mean()
    if mask.sum() == 0:
        zero = torch.tensor(0.0, device=zv_target.device)
        return zero, valid_ratio

    zv_sel = zv_target[mask]
    pseudo_sel = pseudo[mask]
    if metric == "cosine":
        zv = l2_normalize(zv_sel)
        zs = l2_normalize(zs_prototypes)
        logits = torch.matmul(zv, zs.t())
        loss = F.cross_entropy(logits, pseudo_sel)
    elif metric == "mse":
        target_zs = zs_prototypes[pseudo_sel]
        loss = F.mse_loss(zv_sel, target_zs)
    else:
        raise ValueError(f"Unsupported semantic metric: {metric}")

    return loss, valid_ratio
