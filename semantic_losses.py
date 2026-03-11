import torch
import torch.nn.functional as F

from semantic_modules import l2_normalize


def semantic_logits(
    zv: torch.Tensor,
    zs_prototypes: torch.Tensor,
    metric: str = "cosine",
    logit_scale: float = 16.0,
) -> torch.Tensor:
    if metric == "cosine":
        zv_norm = l2_normalize(zv)
        zs_norm = l2_normalize(zs_prototypes)
        return logit_scale * torch.matmul(zv_norm, zs_norm.t())
    if metric == "mse":
        # Negative squared distance as class score.
        zv_sq = (zv ** 2).sum(dim=1, keepdim=True)
        zs_sq = (zs_prototypes ** 2).sum(dim=1, keepdim=True).t()
        dist_sq = zv_sq + zs_sq - 2 * torch.matmul(zv, zs_prototypes.t())
        return -dist_sq
    raise ValueError(f"Unsupported semantic metric: {metric}")


def source_semantic_alignment_loss(
    zv_source: torch.Tensor,
    source_label: torch.Tensor,
    zs_prototypes: torch.Tensor,
    metric: str = "cosine",
    logit_scale: float = 16.0,
) -> torch.Tensor:
    logits = semantic_logits(zv_source, zs_prototypes, metric=metric, logit_scale=logit_scale)
    return F.cross_entropy(logits, source_label)


def target_semantic_consistency_loss(
    target_logits: torch.Tensor,
    semantic_logits_target: torch.Tensor,
    conf_threshold: float = 0.9,
    margin_threshold: float = 0.0,
):
    probs = F.softmax(target_logits, dim=1)
    conf, pseudo = torch.max(probs, dim=1)
    conf_mask = conf >= conf_threshold

    sem_probs = F.softmax(semantic_logits_target, dim=1)
    top2 = torch.topk(sem_probs, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    margin_mask = margin >= margin_threshold

    mask = conf_mask & margin_mask

    valid_ratio = mask.float().mean()
    conf_pass_ratio = conf_mask.float().mean()
    margin_pass_ratio = margin_mask.float().mean()
    if mask.sum() == 0:
        zero = torch.tensor(0.0, device=target_logits.device)
        return zero, conf_pass_ratio, margin_pass_ratio, valid_ratio

    pseudo_sel = pseudo[mask]
    loss = F.cross_entropy(semantic_logits_target[mask], pseudo_sel)

    return loss, conf_pass_ratio, margin_pass_ratio, valid_ratio
