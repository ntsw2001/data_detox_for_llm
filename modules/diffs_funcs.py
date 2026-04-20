
import torch
import geomloss


def fkl_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, **kwargs) -> torch.Tensor:
    fkl_score = torch.nn.functional.kl_div(
        input=toxic_logits.log_softmax(dim=-1),
        target=base_logits.softmax(dim=-1),
        reduction='batchmean',
        log_target=False
    )
    return fkl_score


def rkl_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, **kwargs) -> torch.Tensor:
    rkl_score = torch.nn.functional.kl_div(
        input=base_logits.log_softmax(dim=-1),
        target=toxic_logits.softmax(dim=-1),
        reduction='batchmean',
        log_target=False
    )
    return rkl_score


def js_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, **kwargs) -> torch.Tensor:
    js_score = 0.5 * fkl_diffs_fn(base_logits, toxic_logits) + 0.5 * rkl_diffs_fn(base_logits, toxic_logits)
    
    return js_score


def tvd_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, **kwargs) -> torch.Tensor:
    p = torch.nn.functional.softmax(base_logits, dim=-1)
    q = torch.nn.functional.softmax(toxic_logits, dim=-1)

    # Total Variation Distance
    tvd_score = 0.5 * torch.sum(torch.abs(p - q), dim=-1)
    
    return tvd_score


def wsd_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, wsd_p: int = 1, wsd_blur: float = 0.01) -> torch.Tensor:
    assert len(base_logits.shape) in [2, 3] and len(toxic_logits.shape) in [2, 3]
    
    p = torch.nn.functional.softmax(base_logits, dim=-1)
    q = torch.nn.functional.softmax(toxic_logits, dim=-1)

    loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=wsd_p, blur=wsd_blur)  # p=1 -> Wasserstein-1 距离
    emd = loss_fn(p, q)
    
    return emd


DiffsFuncs = {
    "none": None,
    "fkl": fkl_diffs_fn,
    "rkl": rkl_diffs_fn,
    "wsd": wsd_diffs_fn,
    "js": js_diffs_fn,
    "tvd": tvd_diffs_fn
}
