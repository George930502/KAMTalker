import argparse
import json
import random
import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from datasets import VideoAudioDataset
from model import DiffTalkingHead
from warmup_scheduler import GradualWarmupScheduler


def l2_velocity_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Velocity regularizer (finite difference)."""
    return F.mse_loss(pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])


def l2_acceleration_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Acceleration regularizer."""
    vel_pred = pred[:, 1:] - pred[:, :-1]
    vel_gt = gt[:, 1:] - gt[:, :-1]
    acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
    acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
    return F.mse_loss(acc_pred, acc_gt)


def pad_to_len(t: torch.Tensor, length: int, dim: int, mode: str = 'zero') -> torch.Tensor:
    pad_size = list(t.shape)
    pad_size[dim] = length - t.shape[dim]
    if pad_size[dim] <= 0:
        return t
    if mode == 'zero':
        pad_tensor = torch.zeros(*pad_size, dtype=t.dtype, device=t.device)
    elif mode == 'replicate':
        idx = torch.tensor([t.shape[dim] - 1], device=t.device)
        pad_tensor = t.index_select(dim, idx).expand(*pad_size)
    else:
        raise ValueError(f'Unknown pad_mode: {mode}')
    return torch.cat([t, pad_tensor], dim=dim)


def maybe_truncate(audio: torch.Tensor, motion: torch.Tensor, args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly truncate a window with probability args.trunc_prob."""
    B, Tw = motion.shape[:2]
    if random.random() >= args.trunc_prob:
        return audio, motion, torch.ones(B, Tw, dtype=torch.bool, device=motion.device)

    end_idx = torch.randint(1, Tw, (B,), device=motion.device)
    spp = int(round(args.audio_sr / args.fps))
    out_a, out_m, mask_list = [], [], []

    for b in range(B):
        t_end = end_idx[b].item()
        s_end = t_end * spp
        a_tr = audio[b, :s_end]
        m_tr = motion[b, :t_end]
        a_pad = pad_to_len(a_tr.unsqueeze(0), audio.shape[1], dim=1, mode=args.pad_mode).squeeze(0)
        m_pad = pad_to_len(m_tr.unsqueeze(0), Tw, dim=1, mode=args.pad_mode).squeeze(0)
        out_a.append(a_pad)
        out_m.append(m_pad)
        mask_list.append(torch.arange(Tw, device=audio.device) < t_end)

    return torch.stack(out_a), torch.stack(out_m), torch.stack(mask_list)


def split_into_windows(audio: torch.Tensor, motion: torch.Tensor, args):
    Tw, Tp = args.n_motions, args.n_prev_motions
    spp = int(round(args.audio_sr / args.fps))

    # window 0
    m0_full = motion[:, :Tw]
    a0_full = audio[:, :Tw * spp]

    # window 1 overlaps by Tp
    m1_full = motion[:, Tw - Tp: Tw + Tw - Tp]
    a1_full = audio[:, (Tw * spp) - (Tp * spp): (Tw * spp) * 2 - (Tp * spp)]

    a0, m0, ind0 = maybe_truncate(a0_full, m0_full, args)
    a1, m1, ind1 = maybe_truncate(a1_full, m1_full, args)

    return (a0, m0, ind0), (a1, m1, ind1)


def collect_entries(split_root: Path) -> List[str]:
    return sorted([p.stem.replace('_enhanced_gt', '') for p in split_root.glob('*_enhanced_gt.npz')])


def build_dataset(args, split: str):
    ds = VideoAudioDataset(
        gt_dir=Path(args.gt_root),
        audio_dir=Path(args.audio_dir),
        split=split,
        n_motions=args.n_motions * 2,
        audio_sampling_rate=args.audio_sr,
        video_fps=args.fps,
        crop_strategy=args.crop_strategy
    )
    return ds


def build_loader(args, shuffle: bool, split: str):
    return DataLoader(
        build_dataset(args, split),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True
    )


def window_forward(model, a_pair, m_pair, ind_pair, device, args, train=True):
    model.train() if train else model.eval()

    _, pred0, gt0, audio_feat0 = model(
        motion_feat=m_pair[0].to(device),
        audio_or_feat=a_pair[0].to(device),
        indicator=ind_pair[0].to(device) if args.use_indicator else None
    )

    prev_m = m_pair[0][:, -args.n_prev_motions:].to(device)
    prev_a = audio_feat0[:, -args.n_prev_motions:]

    _, pred1, gt1, _ = model(
        motion_feat=m_pair[1].to(device),
        audio_or_feat=a_pair[1].to(device),
        prev_motion_feat=prev_m,
        prev_audio_feat=prev_a,
        indicator=ind_pair[1].to(device) if args.use_indicator else None
    )

    # MSE
    if args.l_mse > 0:
        loss = args.l_mse * (F.mse_loss(pred0, gt0) + F.mse_loss(pred1, gt1))
        #print(f"l_mse: {loss.item():.4f}")

    # Velocity
    if args.l_vel > 0:
        loss = loss + args.l_vel * (
            l2_velocity_loss(pred0, gt0) + l2_velocity_loss(pred1, gt1)
        )
        #print(f"l_vel: {loss.item():.4f}")

    # Acceleration
    if args.l_acc > 0:
        loss = loss + args.l_acc * (
            l2_acceleration_loss(pred0, gt0) + l2_acceleration_loss(pred1, gt1)
        )
        #print(f"l_acc: {loss.item():.4f}")

    return loss


def main():
    parser = argparse.ArgumentParser(description='Talking Face Video Generation Task: Stage2')

    # data & splitting
    parser.add_argument('--gt_root', required=True)  # /root/video_generation/Stage2/ground_truth
    parser.add_argument('--audio_dir', required=True)  # /root/video_generation/video_enhancement/GFPGAN/extracted_audio
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    # model / diffusion
    parser.add_argument('--target', type=str, default='sample', choices=['noise', 'sample'])
    parser.add_argument('--n_diff_steps', type=int, default=500)
    parser.add_argument('--diff_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
    parser.add_argument('--audio_model', type=str, default='hubert', choices=['hubert'])
    parser.add_argument('--architecture', type=str, default='decoder', choices=['decoder'])
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--align_mask_width', type=int, default=1)
    parser.add_argument('--no_use_learnable_pe', action='store_true')

    # sequence & truncation
    parser.add_argument('--n_motions', type=int, default=45)
    parser.add_argument('--n_prev_motions', type=int, default=10)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--crop_strategy', type=str, default='random', choices=['begin', 'random', 'end'])
    parser.add_argument('--trunc_prob', type=float, default=0.3)
    parser.add_argument('--pad_mode', type=str, choices=['zero', 'replicate'], default='zero')
    parser.add_argument('--use_indicator', type=bool, default=True)

    # training
    # parser.add_argument('--iter', type=int, default=30000, help='total training iterations')
    parser.add_argument('--epochs', type=int, default=400, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--l_mse', type=float, default=1.0, help='weight for l2 loss')
    parser.add_argument('--l_vel', type=float, default=0.8, help='weight for velocity loss')
    parser.add_argument('--l_acc', type=float, default=0.6, help='weight for acceleration loss')
    parser.add_argument('--l_dtw', type=float, default=0.0, help='weight for Soft-DTW loss')
    parser.add_argument('--dtw_gamma', type=float, default=0.0, help='gamma parameter for Soft-DTW')
    # parser.add_argument('--save_every', type=int, default=500, help='save & validate every N iterations')

    # scheduler / warmup
    parser.add_argument('--scheduler', choices=['None', 'Warmup', 'WarmupThenDecay'], default='WarmupThenDecay')
    # parser.add_argument('--warm_iter', type=int, default=250)
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--warmup_multiplier', type=int, default=10, help='Target LR / initial LR during warmup')
    # parser.add_argument('--cos_max_iter', type=int, default=120000)
    parser.add_argument('--cos_max_epochs', type=int, default=20, help='Max total epochs for cosine decay')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01)

    # misc
    parser.add_argument('--audio_sr', type=int, default=16000)
    parser.add_argument('--exp_name', default='run')
    parser.add_argument('--save_dir', default='experiments')

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare experiment directory
    exp_dir = Path(args.save_dir) / f"{args.exp_name}_{datetime.now():%y%m%d_%H%M%S}"
    (exp_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    with open(exp_dir / 'hyperparameters.json', 'w') as fp:
        json.dump(vars(args), fp, indent=2)

    # data loaders
    train_loader = build_loader(args, shuffle=True, split='train')
    val_loader = build_loader(args, shuffle=False, split='val')

    # model & optimizer
    cfg = argparse.Namespace(**vars(args))
    model = DiffTalkingHead(cfg, device=device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr / args.warmup_multiplier,
        weight_decay=args.weight_decay
    )

    # scheduler
    scheduler = None
    if args.scheduler in ('Warmup', 'WarmupThenDecay'):
        if args.scheduler == 'Warmup':
            scheduler = GradualWarmupScheduler(optimizer, multiplier=args.warmup_multiplier, total_epoch=args.warmup_epochs)
        else:
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.cos_max_epochs - args.warmup_epochs,
                eta_min=args.lr * args.min_lr_ratio
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=args.warmup_multiplier,
                total_epoch=args.warmup_epochs,
                after_scheduler=cosine_scheduler
            )

    # infinite train iterator 
    # train_iter = itertools.cycle(train_loader)
    best_val = float('inf')

    # main training loop
    # pbar = tqdm(total=args.iter, desc='Training', unit='iter')

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', unit='batch')

        for audio_raw, motion_raw, _ in pbar:
            (a0, m0, ind0), (a1, m1, ind1) = split_into_windows(audio_raw, motion_raw, args)

            loss = window_forward(model, (a0, a1), (m0, m1), (ind0, ind1), device, args, train=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_lr()[0]:.2e}'
            })

        if scheduler:
            scheduler.step()

        # ===== Save checkpoint after each epoch =====
        # ckpt = exp_dir / 'checkpoints' / f'epoch_{epoch:03d}.pt'
        # torch.save({
        #     'epoch': epoch,
        #     'model': model.state_dict(),
        #     'opt': optimizer.state_dict()
        # }, ckpt)

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audio_v, motion_v, _ in val_loader:
                (va0, vm0, vi0), (va1, vm1, vi1) = split_into_windows(audio_v, motion_v, args)
                val_loss += window_forward(model, (va0, va1), (vm0, vm1), (vi0, vi1), device, args, train=False).item()
        val_loss /= len(val_loader)

        tqdm.write(f"[Epoch {epoch:03d}] val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = exp_dir / 'checkpoints' / 'best.pt'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'val_loss': best_val
            }, best_ckpt)
            tqdm.write(f"→ new best (val={best_val:.6f}), saved best.pt")

    pbar.close()
    print("Training complete.")


if __name__ == '__main__':
    main()
