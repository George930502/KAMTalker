import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from face_alignment import FaceAlignment, LandmarksType
from skimage.util import img_as_float32
from skimage.color import gray2rgb
from imageio import get_reader
from tqdm import tqdm
import argparse
import random

sys.path.append(os.path.abspath('..'))

from modules.Emtn import HeadPoseEstimator_ExpressionDeformationEstimator
from modules.utils import getRotationMatrix

def read_video(path):
    if path.lower().endswith(('.mp4', '.gif')):
        reader = get_reader(path)
        frames = []
        for frame in reader:
            img = img_as_float32(frame)
            if img.ndim == 2:
                img = gray2rgb(img)
            if img.shape[-1] == 4:
                img = img[..., :3]
            frames.append(img)
        return np.stack(frames, axis=0)
    else:
        raise ValueError(f"Unsupported video format: {path}")

def split_dataset(root_dir):  #
    all_videos = [f for f in os.listdir(root_dir) if f.lower().endswith(('.mp4'))]
    random.shuffle(all_videos)

    n_total = len(all_videos)
    n_train = int(n_total * 0.85)
    n_val   = int(n_total * 0.10)

    split_dirs = {'train': all_videos[:n_train],
                  'val': all_videos[n_train: n_train + n_val],
                  'test':all_videos[n_train + n_val:]}

    for split, files in split_dirs.items():
        split_path = os.path.join(root_dir, split)
        os.makedirs(split_path, exist_ok=True)
        for f in files:
            shutil.move(os.path.join(root_dir, f), os.path.join(split_path, f))

class MultiFrameDataset(Dataset):
    def __init__(self, root_dir, mode='train', num_frames=10):
        super().__init__()
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.mode = mode

        expected_dirs = ['train', 'val', 'test']
        if not all(os.path.isdir(os.path.join(root_dir, d)) for d in expected_dirs):
            print("Splitting dataset...")
            split_dataset(root_dir) 

        self.data_dir = os.path.join(root_dir, mode)
        self.videos = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.lower().endswith(('.mp4', '.gif'))]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        path = self.videos[idx]
        vid = read_video(path)                        # (T, H, W, 3)
        T = vid.shape[0]
        indices = np.random.choice(T, self.num_frames, replace=(T < self.num_frames))
        imgs = vid[indices].astype('float32')         # (N, H, W, 3)
        imgs = imgs.transpose(0, 3, 1, 2)             # (N, 3, H, W)
        return torch.from_numpy(imgs)

def extract_keypoints(batch, fa: FaceAlignment):
    device = batch.device
    kps = []
    for img in batch:
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        lms = fa.get_landmarks(arr)
        if not lms:
            raise RuntimeError("No face found")
        kps.append(torch.tensor(lms[0], dtype=torch.float32))
    return torch.stack(kps).to(device)

jaw = [(i, i + 1) for i in range(16)]
left_eye = [(i, i + 1) for i in range(36, 41)] + [(41, 36)]
right_eye = [(i, i + 1) for i in range(42, 47)] + [(47, 42)]
nose_bridge = [(i, i + 1) for i in range(27, 30)]
nose_wings = [(31, 32),(32, 33),(33, 34),(34, 35)]
outer_mouth = [(i, i + 1) for i in range(48, 59)] + [(59, 48)]
inner_mouth = [(i, i + 1) for i in range(60, 67)] + [(67, 60)]

BONE_EDGES = jaw + left_eye + right_eye + nose_bridge + nose_wings + outer_mouth + inner_mouth


def bone_loss_fn(src,tgt): 
    return torch.mean(torch.stack([(tgt[:, j] - tgt[:, i]).norm(dim = -1) - (src[:, j] - src[:, i]).norm(dim = -1) for i, j in BONE_EDGES]))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=r'C:\Users\george\VG_Project\dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='hpe_checkpoint')
    parser.add_argument('--num_frames', type=int, default=10, help="number of frames to extract")
    parser.add_argument('--lambda_bone', type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = MultiFrameDataset(args.root_dir, mode='train', num_frames=args.num_frames)
    val_ds = MultiFrameDataset(args.root_dir, mode='val', num_frames=args.num_frames)
    test_ds = MultiFrameDataset(args.root_dir, mode='test', num_frames=args.num_frames)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # face-alignment
    fa = FaceAlignment(LandmarksType.THREE_D, flip_input=False)

    # head‐pose + expression/deformation 
    NUM_KP = 68
    hpe_model = HeadPoseEstimator_ExpressionDeformationEstimator(num_kp=NUM_KP).to(device)

    optimizer = optim.AdamW(hpe_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    mse_loss = nn.MSELoss()
    best_val = float('inf')

    for epoch in range(1, args.epochs+1):
        hpe_model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            # batch: (B, N, 3, H, W)
            B, N, C, H, W = batch.shape
            imgs = batch.view(B*N, C, H, W).to(device)  # (B*N,3,H,W)

            # 1) 3D keypoints
            try:
                kp = extract_keypoints(imgs, fa)         # (B*N, K, 3)
            except RuntimeError:
                continue

            # 2) HeadPoseEstimator +
            yaw, pitch, roll, translation, deformation = hpe_model(imgs)
            # yaw,pitch,roll: (B*N,)
            # translation: (B*N,3)
            # deformation: (B*N, K, 3)

            # 3) 反推 canonical keypoints
            R = getRotationMatrix(yaw, pitch, roll)      # (B*N,3,3)
            invR = R.transpose(1,2)                      # (B*N,3,3)
            # (kp - trans - deform) @ invR
            canon = torch.matmul(
                kp - translation.unsqueeze(1) - deformation,
                invR
            )                                             # (B*N, K, 3)    
            # bone loss
            bone_loss = bone_loss_fn(kp, canon)

            canon = canon.view(B, N, -1, 3)               # (B, N, K, 3)

            # 4) 同影片內 N frames canonical keypoints 應一致
            canon_mean = canon.mean(dim=1, keepdim=True)  # (B,1,K,3)
            loss_cons = mse_loss(canon, canon_mean.expand_as(canon))

            loss = loss_cons + args.lambda_bone * bone_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation
        hpe_model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                B, N, C, H, W = batch.shape
                imgs = batch.view(B*N, C, H, W).to(device)
                try:
                    kp = extract_keypoints(imgs, fa)
                except RuntimeError:
                    continue

                yaw, pitch, roll, translation, deformation = hpe_model(imgs)
                R = getRotationMatrix(yaw, pitch, roll)
                invR = R.transpose(1,2)
                canon = torch.matmul(kp - translation.unsqueeze(1) - deformation, invR)
                # bone loss
                bone_loss = bone_loss_fn(kp, canon)
                # consistency loss across frames
                canon = canon.view(B, N, -1, 3)
                canon_mean = canon.mean(dim=1, keepdim=True)
                loss_cons = mse_loss(canon, canon_mean.expand_as(canon))

                loss_val = loss_cons + args.lambda_bone * bone_loss

                val_loss += loss_val.item()
                val_count += 1

        avg_tr = total_loss / len(train_loader)
        avg_vl = val_loss / max(val_count, 1)
        scheduler.step(avg_vl)

        print(f"[Epoch {epoch}] Train Loss: {avg_tr:.6f} | Val Loss: {avg_vl:.6f}")

        if avg_vl < best_val:
            best_val = avg_vl
            torch.save(hpe_model.state_dict(), os.path.join(args.save_dir, 'best_hpe.pth'))

    torch.save(hpe_model.state_dict(), os.path.join(args.save_dir, 'final_hpe.pth'))
    print("Training completed.")
