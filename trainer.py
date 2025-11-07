import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as udata
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import pandas as pd
import wandb
from tqdm import trange, tqdm
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

from data import DementiaDetectionDataset, DementiaPredictionDataset
from models.encoder import build_model


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class Trainer:
    def __init__(self, args):
        self._init_distributed_mode()
        self.args = args
        self.desc = f'{args.arch}_{args.ft}'
        if args.ft == 'lora':
            self.desc += f'_rank_{args.lora_rank}_ft_{args.ft_blks}'
        elif args.ft == 'partial':
            self.desc += f'_ft_{args.ft_blks}'
        
        self._build_loader()
        self._build_model()
        self._build_optimizer()
        
        self._init_file()
        self._init_wandb()
        
    def _init_distributed_mode(self):
        dist.init_process_group(backend="nccl")
        dist.barrier()

        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.cuda.current_device()
        
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
    def _build_loader(self):
        dist_sampler = partial(udata.distributed.DistributedSampler, 
                               num_replicas=self.world_size, rank=self.global_rank)
        self.train_dataset = DementiaDetectionDataset('train', self.args.img_size)
        self.train_sampler = dist_sampler(dataset=self.train_dataset, shuffle=True, drop_last=True)
        self.train_loader = udata.DataLoader(
            dataset=self.train_dataset,
            sampler=self.train_sampler,
            pin_memory=True,
            drop_last=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        self.valid_dataset = DementiaDetectionDataset('valid', self.args.img_size)
        self.valid_sampler = dist_sampler(dataset=self.valid_dataset, shuffle=False)
        self.test_dataset = DementiaDetectionDataset('test', self.args.img_size)
        self.test_sampler = dist_sampler(dataset=self.test_dataset, shuffle=False)
        self.valid_loader = udata.DataLoader(
            dataset=self.valid_dataset,
            sampler=self.valid_sampler,
            pin_memory=True,
            batch_size=8,
            num_workers=4,
        )
        self.test_loader = udata.DataLoader(
            dataset=self.test_dataset,
            sampler=self.test_sampler,
            pin_memory=False,
            batch_size=8,
            num_workers=4,
        )
        
        self.prediction_dataset = DementiaPredictionDataset(self.args.img_size)
        self.prediction_sampler = dist_sampler(dataset=self.prediction_dataset, shuffle=False)
        self.prediction_loader = udata.DataLoader(
            dataset=self.prediction_dataset,
            sampler=self.prediction_sampler,
            pin_memory=False,
            batch_size=8,
            num_workers=4,
        )
    
    def _build_model(self):
        self.model = DistributedDataParallel(
            build_model(self.args).cuda(),
            device_ids=[self.device]
        )
        self.criterion = nn.BCEWithLogitsLoss()
    
    def _build_optimizer(self):
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.enable_amp = self.args.enable_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)
    
    def _init_file(self):
        root = Path(self.args.root_dir)
        self.log_dir = root/self.desc
        if (self.global_rank == 0) and (not self.log_dir.exists()):
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
    def _init_wandb(self):
        if self.args.wandb and self.global_rank==0:
            wandb.init(
                entity=self.args.wandb_entity,
                project=self.args.wandb_project,
                name=self.desc,
            )
            wandb.run.save('/home/hch/dementia/wandb')
    
    def save(self):
        if self.global_rank == 0:
            torch.save(self.model.module.state_dict(), self.log_dir/'ckpt.pth.tar')
        dist.barrier()
    
    def load(self):
        ckpt = torch.load(self.log_dir/'ckpt.pth.tar', map_location=f'cuda:{self.device}')
        self.model.module.load_state_dict(ckpt)
    
    def train_one_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.model.train()
        running_loss = torch.tensor([0.], device=self.device)
        
        for x, y in tqdm(self.train_loader, desc='train', total=len(self.train_loader), disable=self.global_rank!=0):
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=self.enable_amp):
                x = x.cuda(self.local_rank, non_blocking=True)
                y = y.cuda(self.local_rank, non_blocking=True)

                out = self.model(x)
                loss = self.criterion(out.squeeze(), y)
                running_loss += loss.detach().clone().view(1)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        mean_loss = (concat_all_gather(running_loss).mean() / len(self.train_loader)).item()
        if self.global_rank == 0:
            print(f'Epoch {epoch}: train loss {mean_loss:.3f}')
            if self.args.wandb:
                wandb.log({'train_loss': mean_loss}, step=epoch)
    

    @torch.no_grad()
    def valid_one_epoch(self, epoch=1, kind='valid'):
        from sklearn.metrics import roc_curve
    
        self.model.eval()
        loader  = self.valid_loader if kind == 'valid' else self.test_loader
        sampler = self.valid_sampler if kind == 'valid' else self.test_sampler
    
        preds, targets = [], []
        sampler.set_epoch(0)
    
        for x, y in tqdm(loader, desc=kind, total=len(loader), disable=self.global_rank!=0):
            x = x.cuda(self.local_rank, non_blocking=True)
            y = y.cuda(self.local_rank, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=self.enable_amp):
                p = torch.sigmoid(self.model(x))
            preds.append(concat_all_gather(p))
            targets.append(concat_all_gather(y))
    
        preds   = torch.cat(preds).detach().cpu().squeeze().numpy()
        targets = torch.cat(targets).detach().cpu().squeeze().numpy().astype(int)
    
        preds   = preds.reshape(-1)
        targets = targets.reshape(-1)
    
        try:
            auc = roc_auc_score(targets, preds)
        except ValueError:
            auc = float('nan')
    
        if kind == 'valid':
            try:
                fpr, tpr, thr = roc_curve(targets, preds)
                youden = tpr - fpr
                best_idx  = youden.argmax()
                best_thr  = float(thr[best_idx])
                best_sens = float(tpr[best_idx])
                best_spec = float(1.0 - fpr[best_idx])
    
                self.best_threshold_valid = best_thr
                self.best_sens_valid = best_sens
                self.best_spec_valid = best_spec
    
                if self.global_rank == 0:
                    print(f"\tvalid auc {auc:.3f}")
                    print(f"\tvalid Youden's J best threshold: {best_thr:.6f} "
                          f"(sensitivity={best_sens:.3f}, specificity={best_spec:.3f})")
                    if self.args.wandb:
                        wandb.log({
                            'valid_auc': auc,
                            'youden_best_threshold_valid': best_thr,
                            'youden_best_sensitivity_valid': best_sens,
                            'youden_best_specificity_valid': best_spec
                        }, step=epoch)
            except ValueError:
                if self.global_rank == 0:
                    print(f"\tvalid auc {auc:.3f}")
                    print("\tvalid Youden's J: cannot compute (single-class labels)")
                    if self.args.wandb:
                        wandb.log({'valid_auc': auc}, step=epoch)
            return auc

        if self.global_rank == 0:
            print(f'\t{kind} auc {auc:.3f}')
            if self.args.wandb:
                if kind == 'test':
                    wandb.log({'test_auc': auc})
                else:
                    wandb.log({f'{kind}_auc': auc}, step=epoch)
        return auc


    

    @torch.no_grad()
    def prediction(self):
        """
        Predict on prediction_loader, compute C-index, and print the threshold
        that maximizes Youden's J (TPR - FPR) using events as binary labels.
        """
        from sklearn.metrics import roc_curve

        self.model.eval()

        preds, events, times = [], [], []
        for x, e, t in tqdm(self.prediction_loader, desc='prediction', total=len(self.prediction_loader), disable=self.global_rank!=0):
            x = x.cuda(self.local_rank, non_blocking=True)
            e = e.cuda(self.local_rank, non_blocking=True)
            t = t.cuda(self.local_rank, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=self.enable_amp):
                p = torch.sigmoid(self.model(x))
            preds.append(concat_all_gather(p))
            events.append(concat_all_gather(e))
            times.append(concat_all_gather(t))

        preds = torch.cat(preds).cpu().squeeze().numpy()   # risk scores (higher = riskier)
        events = torch.cat(events).cpu().squeeze().numpy() # binary labels (0/1)
        times  = torch.cat(times).cpu().squeeze().numpy()  # survival times

        # ---- Youden J threshold (on events vs preds) ----
        # roc_curve returns thresholds (including +inf at index 0)
        fpr, tpr, thr = roc_curve(events.astype(int), preds)
        youden = tpr - fpr
        best_idx = youden.argmax()
        best_thr = float(thr[best_idx])
        best_sens = float(tpr[best_idx])
        best_spec = float(1.0 - fpr[best_idx])

        # ---- C-index (time-to-event) ----
        cindex = concordance_index(
            event_times=times,
            predicted_scores=-preds,     # higher risk -> shorter survival
            event_observed=events
        )

        if self.global_rank == 0:
            print(f"Prediction — C-index: {cindex:.3f}")
            print(f"Prediction — Youden's J best threshold: {best_thr:.6f} "
                  f"(sensitivity={best_sens:.3f}, specificity={best_spec:.3f})")
            if self.args.wandb:
                wandb.log({
                    'test_cindex': cindex,
                    'youden_best_threshold_prediction': best_thr,
                    'youden_best_sensitivity_prediction': best_sens,
                    'youden_best_specificity_prediction': best_spec
                })

        # keep return signature the same
        # (inference() expects a single float)
        # also store for later use if needed
        self.best_threshold_prediction = best_thr
        return cindex

        
    # -------------------------
    # ✅ Modified: Early stopping + LR halving
    # -------------------------
    def train(self):
        best_auc = 0.
        stalls = 0

        for epoch in trange(self.args.epochs, desc='Epochs', disable=self.global_rank!=0):
            self.train_one_epoch(epoch)
            auc = self.valid_one_epoch(epoch, 'valid')
            improved = auc > (best_auc + 5e-5)
            if improved:
                best_auc = auc
                stalls = 0
                self.save()
            else:
                stalls += 1
                if stalls == self.args.lr_halve_patience:
                    for g in self.optimizer.param_groups:
                        g["lr"] *= 0.5
                    if self.global_rank == 0:
                        print("  ↳ LR halved")
                if stalls >= self.args.early_stop_patience:
                    if self.global_rank == 0:
                        print("  ↳ Early stop.")
                    break


    def inference(self):
        self.load()
        auc = self.valid_one_epoch(self.args.epochs, 'test')
        cindex = self.prediction()
        pd.DataFrame({'auc': [auc], 'cindex': [cindex]}).to_csv(self.log_dir/'res.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dementia Detection Trainer')
    parser.add_argument('--arch', type=str, 
                        choices=['retfound', 'mae', 'openclip', 'dinov2', 'dinov3', 'retfound_dinov2'])
    parser.add_argument('--ft', type=str, 
                        choices=['linear', 'partial', 'lora'])
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--ft_blks', type=str, default='1',
                    help='LoRA block number or "full"')
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--enable_amp', action='store_true')
    parser.add_argument('--root_dir', type=str, default='/home/kjw/Projects/dementia/ckpts')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default='kiimmm')
    parser.add_argument('--wandb_project', type=str, default='dementia')

    # ✅ Added arguments for early stopping strategy
    parser.add_argument('--lr_halve_patience', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    
    args = parser.parse_args()

    if isinstance(args.ft_blks, str) and args.ft_blks.isdigit():
        args.ft_blks = int(args.ft_blks)
    else:
        if isinstance(args.ft_blks, str):
            args.ft_blks = args.ft_blks.lower()
    
    trainer = Trainer(args)
    trainer.train()
    trainer.inference()
