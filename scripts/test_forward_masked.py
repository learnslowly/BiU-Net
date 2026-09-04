"""Semantic check: HybridFocalLoss.forward_masked should match the original
forward() called on boolean-indexed slices, up to floating-point tolerance.
"""
import sys
import torch
sys.path.insert(0, ".")
from config.modelconfig import ModelConfig
from data.utils import HybridFocalLoss


def main():
    torch.manual_seed(0)
    cfg = ModelConfig(
        runId="semchk", dataset="LOS", chromosome=22, segLen=128, overlap=16,
        population="ALL", model="unet", loss="hybridWeightedFocalLoss",
        dosageLossLambda=0.1, focalGamma=2.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crit = HybridFocalLoss(cfg).to(device)

    B, S, V = 8, 128, 6
    logits = torch.randn(B, S, V, device=device, dtype=torch.float32)
    # Labels: 1..vocab-2 valid genotypes, plus some pad positions (padLabel=-3)
    labels_raw = torch.randint(1, V - 1, (B, S), device=device)
    # Sprinkle some "padding" positions (padLabel = -3 in this codebase)
    pad_mask = torch.rand(B, S, device=device) > 0.1  # 90% valid
    pad_label_const = -3
    labels = torch.where(pad_mask, labels_raw, torch.full_like(labels_raw, pad_label_const))

    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.flatten()
    pad_mask_flat = pad_mask.flatten()
    valid_positions = pad_mask_flat  # benchmarkAll=True: loss_mask = valid_positions

    # ---- Old path (variable shape) ----
    logits_valid = logits_flat[valid_positions]
    labels_valid = labels_flat[valid_positions]
    loss_old = crit(logits_valid, labels_valid)

    # ---- New path (constant shape) ----
    loss_new = crit.forward_masked(logits_flat, labels_flat, valid_positions)

    print(f"loss_old = {loss_old.item():.10f}")
    print(f"loss_new = {loss_new.item():.10f}")
    assert torch.allclose(loss_old, loss_new, atol=1e-5, rtol=1e-5), (
        f"loss mismatch: old={loss_old.item()} new={loss_new.item()}"
    )

    # ---- batch_correct semantic check ----
    pred_raw = logits.argmax(dim=2).flatten()
    pad_id_const = torch.full((), pad_label_const, dtype=pred_raw.dtype, device=device)
    pred_flat = torch.where(pad_mask_flat, pred_raw, pad_id_const)

    # Old way:
    batch_correct_old = (pred_flat[valid_positions] == labels_valid).sum()
    # New way:
    batch_correct_new = ((pred_flat == labels_flat) & valid_positions).sum()
    assert batch_correct_old.item() == batch_correct_new.item(), (
        f"batch_correct mismatch: old={batch_correct_old.item()} "
        f"new={batch_correct_new.item()}"
    )

    batch_total_old = valid_positions.sum()
    batch_total_new = valid_positions.sum()
    assert batch_total_old.item() == batch_total_new.item()

    print("PASS: forward_masked semantics match old path within 1e-5 tolerance.")
    print(f"  batch_correct={batch_correct_new.item()}/{batch_total_new.item()}")


if __name__ == "__main__":
    main()
