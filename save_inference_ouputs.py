'''
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,9

torchrun --nproc_per_node=8 20251013_eval_dinov3etc.py \
  --ckpt_root /home/kjw/Projects/dementia/ckpts \
  --work_dir  /home/hch/dementia \
  --out_dir   /home/hch/dementia/infer_out \
  --img_size  448 --enable_amp

'''
# !/usr/bin/env python
# ddp_infer_save.py
# ============================================================
# DDP inference with order restoration:
#  - Loads ONLY the first (lexicographically) checkpoint under --ckpt_root/*/ckpt.pth.tar
#  - Runs classification test set + survival prediction set
#  - Gathers (idx, preds, labels/events/time) and restores original Dataset order
#  - Handles DDP sampler padding by deduplicating idx and clipping to dataset length
#  - Saves CSVs under: {--out_dir}/{model_desc}/
# ============================================================

import os
import re
import glob
import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data as udata
import pandas as pd
import numpy as np
from tqdm import tqdm

# === Project modules ===
# - data.py should define DementiaDetectionDataset, DementiaPredictionDataset
# - models/encoder.py should define build_model(args) consistent with training
from data import DementiaDetectionDataset, DementiaPredictionDataset
from models.encoder import build_model

warnings.filterwarnings("ignore")


# -------------------------------
# DDP helpers
# -------------------------------
def init_distributed():
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    return world_size, global_rank, local_rank, device


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather then cat along dim=0."""
    world_size = dist.get_world_size()
    tensors_gather = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


# -------------------------------
# Dataset wrapper: return (idx, *original)
# -------------------------------
class IndexWrapDataset(udata.Dataset):
    """Wrap a base dataset to also return the sample index."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # Expect:
        #  - DementiaDetectionDataset(kind="test"): (img, label)
        #  - DementiaPredictionDataset(): (img, event, obs_time)
        return (idx, *item)


# -------------------------------
# Model desc parser (mirrors trainer.py naming)
# -------------------------------
def parse_desc(desc: str):
    """
    Parse directory name to reconstruct model args.
    Supported patterns:
      - {arch}_{ft}
      - {arch}_lora_rank_{r}_ft_{b}
      - {arch}_partial_ft_{b}
      - {arch}_linear
    arch in {retfound, mae, openclip, dinov2, dinov3, retfound_dinov2}
    """
    arch_pat = r"(retfound|mae|openclip|dinov2|dinov3|retfound_dinov2)"

    m_lora = re.match(
        rf"^{arch_pat}_lora_rank_(?P<rank>\d+)_ft_(?P<ft>(?:\d+|full))$",
        desc
    )
    if m_lora:
        arch = m_lora.group(1)  
        rnk = int(m_lora.group("rank"))
        ft_token = m_lora.group("ft")  
        ft_blks = int(ft_token) if ft_token.isdigit() else "full"
        return SimpleNamespace(arch=arch, ft="lora", lora_rank=rnk, ft_blks=ft_blks)

    m_partial = re.match(fr"^{arch_pat}_partial_ft_(\d+)$", desc)
    if m_partial:
        arch, blks = m_partial.group(1), int(m_partial.group(2))
        return SimpleNamespace(arch=arch, ft="partial", lora_rank=4, ft_blks=blks)

    m_linear = re.match(fr"^{arch_pat}_linear$", desc)
    if m_linear:
        arch = m_linear.group(1)
        return SimpleNamespace(arch=arch, ft="linear", lora_rank=4, ft_blks=0)

    # Fallback: {arch}_{ft}
    m_base = re.match(fr"^{arch_pat}_(\w+)$", desc)
    if m_base:
        arch, ft = m_base.group(1), m_base.group(2)
        return SimpleNamespace(arch=arch, ft=ft, lora_rank=4, ft_blks=0)

    raise ValueError(f"Unrecognized model desc format: {desc}")



# -------------------------------
# Data loaders (no shuffle; DistributedSampler pads)
# -------------------------------
def build_loaders(img_size: int, world_size: int, global_rank: int):
    sampler_ctor = udata.distributed.DistributedSampler

    # Classification (test split)
    test_base = DementiaDetectionDataset(kind="test", img_sz=img_size)
    test_ds = IndexWrapDataset(test_base)
    test_sampler = sampler_ctor(test_ds, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=False)
    test_loader = udata.DataLoader(
        dataset=test_ds,
        sampler=test_sampler,
        batch_size=4,
        num_workers=8,
        pin_memory=False,
        persistent_workers=False,
    )

    # Survival prediction (full cohort for risk prediction)
    surv_base = DementiaPredictionDataset(img_sz=img_size)
    surv_ds = IndexWrapDataset(surv_base)
    surv_sampler = sampler_ctor(surv_ds, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=False)
    surv_loader = udata.DataLoader(
        dataset=surv_ds,
        sampler=surv_sampler,
        batch_size=4,
        num_workers=8,
        pin_memory=False,
        persistent_workers=False,
    )
    return (test_loader, test_sampler, test_base), (surv_loader, surv_sampler, surv_base)


# -------------------------------
# Checkpoint discovery
# -------------------------------
def find_first_ckpt(ckpt_root: Path) -> Path:
    """Return the first (lexicographically) ckpt.pth.tar under ckpt_root/*/."""
    candidates = sorted(Path(p) for p in glob.glob(str(ckpt_root / "*" / "ckpt.pth.tar")))
    if not candidates:
        raise FileNotFoundError(f"No ckpt.pth.tar found under {ckpt_root}")
    return candidates[0]


# -------------------------------
# Order-restore utilities
# -------------------------------
def dedup_by_idx(idx_np: np.ndarray, *arrays_np: np.ndarray, target_len: int):
    """
    Given padded idx (sorted), drop duplicates and clip to target_len.
    Assumes idx_np is already sorted stably with argsort(kind='stable').
    Returns: (idx_out, arrays_out...)
    """
    keep = np.ones_like(idx_np, dtype=bool)
    if len(idx_np) > 1:
        keep[1:] = idx_np[1:] != idx_np[:-1]

    idx_out = idx_np[keep]
    arrays_out = [arr[keep] for arr in arrays_np]

    # Clip to original dataset length
    idx_out = idx_out[:target_len]
    arrays_out = [arr[:target_len] for arr in arrays_out]

    assert len(idx_out) == target_len, f"expected {target_len}, got {len(idx_out)}"
    for arr in arrays_out:
        assert len(arr) == target_len
    return (idx_out, *arrays_out)


# -------------------------------
# Inference for one model checkpoint
# -------------------------------
@torch.no_grad()
def run_inference_one_model(args, device, world_size, global_rank, local_rank, ckpt_path: Path):
    # Parse desc to reconstruct model
    desc = ckpt_path.parent.name
    conf = parse_desc(desc)
    model_args = SimpleNamespace(
        arch=conf.arch,
        ft=conf.ft,
        lora_rank=getattr(conf, "lora_rank", 4),
        ft_blks=getattr(conf, "ft_blks", 0),
        img_size=args.img_size,
        enable_amp=args.enable_amp,
    )

    # Build & load model
    model = build_model(model_args).cuda()
    model = DDP(model, device_ids=[device])
    state = torch.load(ckpt_path, map_location=f"cuda:{device}")
    model.module.load_state_dict(state)
    model.eval()

    # Build loaders
    (test_loader, test_sampler, test_base), (surv_loader, surv_sampler, surv_base) = \
        build_loaders(args.img_size, world_size, global_rank)

    # ===== Classification inference (test split) =====
    test_sampler.set_epoch(0)
    all_idx_cls, all_pred_cls, all_tgt_cls = [], [], []

    pbar = tqdm(test_loader, desc=f"[{desc}] test", disable=(global_rank != 0))
    for batch in pbar:
        idx, x, y = batch
        x = x.cuda(local_rank, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=args.enable_amp):
            logits = model(x)                   # shape [B, 1] or [B]
            probs = torch.sigmoid(logits).view(-1)

        # Gather
        idx_g = concat_all_gather(idx.to(torch.int64).cuda(local_rank))
        prd_g = concat_all_gather(probs.detach().to(torch.float32).unsqueeze(1).cuda(local_rank)).squeeze(1)
        tgt_g = concat_all_gather(y.to(torch.float32).cuda(local_rank))

        all_idx_cls.append(idx_g.cpu())
        all_pred_cls.append(prd_g.cpu())
        all_tgt_cls.append(tgt_g.cpu())

    if all_idx_cls:
        all_idx_cls = torch.cat(all_idx_cls).numpy()
        all_pred_cls = torch.cat(all_pred_cls).numpy()
        all_tgt_cls  = torch.cat(all_tgt_cls).numpy()
    else:
        all_idx_cls = all_pred_cls = all_tgt_cls = None

    # ===== Survival prediction inference (full set) =====
    surv_sampler.set_epoch(0)
    all_idx_surv, all_pred_surv, all_event_surv, all_time_surv = [], [], [], []

    pbar2 = tqdm(surv_loader, desc=f"[{desc}] surv", disable=(global_rank != 0))
    for batch in pbar2:
        idx, x, e, t = batch
        x = x.cuda(local_rank, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=args.enable_amp):
            probs = torch.sigmoid(model(x)).view(-1)

        idx_g = concat_all_gather(idx.to(torch.int64).cuda(local_rank))
        prd_g = concat_all_gather(probs.detach().to(torch.float32).unsqueeze(1).cuda(local_rank)).squeeze(1)
        e_g   = concat_all_gather(e.to(torch.float32).cuda(local_rank))
        t_g   = concat_all_gather(t.to(torch.float32).cuda(local_rank))

        all_idx_surv.append(idx_g.cpu())
        all_pred_surv.append(prd_g.cpu())
        all_event_surv.append(e_g.cpu())
        all_time_surv.append(t_g.cpu())

    if all_idx_surv:
        all_idx_surv   = torch.cat(all_idx_surv).numpy()
        all_pred_surv  = torch.cat(all_pred_surv).numpy()
        all_event_surv = torch.cat(all_event_surv).numpy()
        all_time_surv  = torch.cat(all_time_surv).numpy()
    else:
        all_idx_surv = all_pred_surv = all_event_surv = all_time_surv = None

    # ===== Rank 0: restore order & save =====
    dist.barrier()
    if global_rank == 0:
        out_dir = Path(args.out_dir) / desc
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Classification (test) ---
        if all_idx_cls is not None:
            N_test = len(test_base)  # true dataset length (e.g., 2857)
            order = all_idx_cls.argsort(kind="stable")
            idx_sorted  = all_idx_cls[order]
            pred_sorted = all_pred_cls[order]
            tgt_sorted  = all_tgt_cls[order]

            # Deduplicate padding and clip
            idx_final, pred_final, tgt_final = dedup_by_idx(
                idx_sorted, pred_sorted, tgt_sorted, target_len=N_test
            )

            cls_out = pd.DataFrame({
                "idx": idx_final,
                "label": tgt_final,
                "pred": pred_final,
            })
            cls_out.to_csv(out_dir / "test_preds.csv", index=False)

        # --- Survival (prediction) ---
        if all_idx_surv is not None:
            N_surv = len(surv_base)
            order_s = all_idx_surv.argsort(kind="stable")
            idx_s   = all_idx_surv[order_s]
            p_s     = all_pred_surv[order_s]
            e_s     = all_event_surv[order_s]
            t_s     = all_time_surv[order_s]

            idx_final, p_final, e_final, t_final = dedup_by_idx(
                idx_s, p_s, e_s, t_s, target_len=N_surv
            )

            surv_out = pd.DataFrame({
                "idx": idx_final,
                "pred": p_final,
                "event": e_final,
                "obs_time": t_final,
            })
            surv_out.to_csv(out_dir / "prediction_preds.csv", index=False)

        with open(out_dir / "README.txt", "w") as f:
            f.write(
                "Files:\n"
                "- test_preds.csv: [idx, label, pred] (order restored; DDP padding removed)\n"
                "- prediction_preds.csv: [idx, pred, event, obs_time] (order restored; DDP padding removed)\n"
                "Notes:\n"
                "  - We do not re-read CSVs; outputs are built solely from gathered tensors\n"
                "  - DistributedSampler padding handled by idx de-duplication and clipping\n"
            )

    dist.barrier()


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="DDP inference with order-preserving save")
    parser.add_argument("--ckpt_root", type=str, required=True, help="e.g., /home/kjw/Projects/dementia/ckpts")
    parser.add_argument("--work_dir",  type=str, required=True, help="e.g., /home/hch/dementia")
    parser.add_argument("--out_dir",   type=str, required=True, help="e.g., /home/hch/dementia/infer_out")
    parser.add_argument("--img_size",  type=int, default=448)
    parser.add_argument("--enable_amp", action="store_true")
    args = parser.parse_args()

    # Ensure relative reads in data.py work as in training
    os.chdir(args.work_dir)

    world_size, global_rank, local_rank, device = init_distributed()

    # First checkpoint only (lexicographic)
    # ckpt_path = find_first_ckpt(Path(args.ckpt_root))
    # if global_rank == 0:
    #     print(f"[INFO] Using checkpoint: {ckpt_path}")

    # ---- Run single model ----
    # run_inference_one_model(args, device, world_size, global_rank, local_rank, ckpt_path)

    # ---- (Optional) Loop selected models under --ckpt_root ----
    dist.barrier()
    all_ckpts = sorted(Path(p) for p in glob.glob(str(Path(args.ckpt_root) / "*" / "ckpt.pth.tar")))
    
    #selected_ckpts = [p for p in all_ckpts if "full" in p.parent.name]
    #selected_ckpts = [p for p in all_ckpts if "openclip_lora_rank_4_ft_full" in p.parent.name]
    #selected_ckpts = [p for p in all_ckpts if "retfound_lora_rank_4_ft_full" in p.parent.name]
    #selected_ckpts = [p for p in all_ckpts if "retfound_dinov2_lora_rank_4_ft_full" in p.parent.name]
    #selected_ckpts = [p for p in all_ckpts if "mae_lora_rank_4_ft_full" in p.parent.name]

    selected_ckpts = all_ckpts
    
    # if global_rank == 0:
    #     if not selected_ckpts:
    #         raise FileNotFoundError("No checkpoints matched condition: directory name contains 'full'.")
    #     print(f"[INFO] Selected {len(selected_ckpts)} checkpoints to evaluate:")
    #     for p in selected_ckpts:
    #         print("  -", p)
    #     print(f"[INFO] Using first selected checkpoint: {selected_ckpts[0]}")
    
    for path in selected_ckpts:
        run_inference_one_model(args, device, world_size, global_rank, local_rank, path)
    
    if global_rank == 0:
        print("[DONE] Inference finished.")


if __name__ == "__main__":
    main()
