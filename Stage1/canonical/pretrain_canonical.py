import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from face_alignment import FaceAlignment, LandmarksType
from sklearn.model_selection import train_test_split
from skimage.util import img_as_float32
from skimage.color import gray2rgb
from imageio import get_reader
from tqdm import tqdm
import argparse

# ----- 0. 讀取影片函式 -----
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

# ----- 1. 單張影像資料集 -----
class SingleFrameDataset(Dataset):
    def __init__(self, root_dir, is_train=True, random_seed=0, test_size=0.1):
        super().__init__()
        all_items = os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, 'train')):
            self.train_dir = os.path.join(root_dir, 'train')
            self.test_dir  = os.path.join(root_dir, 'test')
        else:
            train_items, test_items = train_test_split(all_items, random_state=random_seed, test_size=test_size)
            self.train_dir = os.path.join(root_dir, 'train')
            self.test_dir  = os.path.join(root_dir, 'test')
            os.makedirs(self.train_dir, exist_ok=True)
            os.makedirs(self.test_dir,  exist_ok=True)
            for v in train_items:
                shutil.move(os.path.join(root_dir, v), self.train_dir)
            for v in test_items:
                shutil.move(os.path.join(root_dir, v), self.test_dir)
        self.is_train = is_train

    def _list_videos(self):
        base = self.train_dir if self.is_train else self.test_dir
        return [os.path.join(base, f) for f in os.listdir(base)]

    def __len__(self):
        return len(self._list_videos())

    def __getitem__(self, idx):
        path = self._list_videos()[idx]
        vid = read_video(path)
        n = vid.shape[0]
        frame_idx = np.random.randint(n)
        img = vid[frame_idx].astype('float32')
        return img.transpose(2,0,1)

# ----- 2. 殘差 MLP Block -----
class ResMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + x)

# ----- 3. CanonicalPointNet -----
class CanonicalPointNet(nn.Module):
    def __init__(self, point_dim=3, emb_dim=64, global_dim=256):
        super().__init__()
        self.init_embed = nn.Sequential(
            nn.Linear(point_dim, emb_dim),
            nn.ReLU(),
            ResMLP(emb_dim),
            ResMLP(emb_dim)
        )
        self.global_net  = nn.Sequential(
            nn.Linear(emb_dim, global_dim), 
            nn.ReLU(), 
            nn.Linear(global_dim, global_dim)
        )
        self.global_skip = nn.Linear(emb_dim, global_dim)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim + global_dim, global_dim),
            nn.ReLU(), 
            nn.Linear(global_dim, point_dim)
        )

    def forward(self, kp):
        B, N, _ = kp.shape
        f = self.init_embed(kp)
        g, _ = f.max(dim = 1)
        g_feat = self.global_net(g) + self.global_skip(g)
        g_exp = g_feat.unsqueeze(1).expand(-1, N, -1)
        feat = torch.cat([f, g_exp], dim=2)
        return self.decoder(feat)

# ----- 4. 隨機旋轉矩陣 -----
def random_rotation_matrix(B, device):
    mats = []
    for _ in range(B):
        A = torch.randn(3, 3, device=device)
        Q, _ = torch.linalg.qr(A, mode='reduced')
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        mats.append(Q)
    return torch.stack(mats)

# ----- 5. Procrustes Align -----
def rigid_procrustes_alignment(X, Y):
    B, N, _ = X.shape
    muX = X.mean(1,keepdim = True)
    muY = Y.mean(1,keepdim = True)
    X0 = X - muX
    Y0 = Y - muY
    C = torch.matmul(Y0.transpose(1,2), X0)
    U, S, Vt = torch.linalg.svd(C)
    det = torch.det(U @ Vt)
    D = torch.eye(3,device = X.device).unsqueeze(0).repeat(B, 1, 1)
    D[:, 2, 2] = torch.sign(det)
    R = U @ D @ Vt
    return (Y0 @ R.transpose(1,2)) + muX

# ----- 6. Bone Loss -----
jaw = [(i, i + 1) for i in range(16)]
left_eye = [(i, i + 1) for i in range(36, 41)] + [(41, 36)]
right_eye = [(i, i + 1) for i in range(42, 47)] + [(47, 42)]
nose_bridge = [(i, i + 1) for i in range(27, 30)]
nose_wings = [(31, 32),(32, 33),(33, 34),(34, 35)]
outer_mouth = [(i, i + 1) for i in range(48, 59)]+[(59, 48)]
inner_mouth = [(i, i + 1) for i in range(60, 67)]+[(67, 60)]

BONE_EDGES = jaw + left_eye + right_eye + nose_bridge + nose_wings + outer_mouth + inner_mouth

def bone_loss_fn(src,tgt): 
    return torch.mean(torch.stack([(tgt[:, j] - tgt[:, i]).norm(dim = -1) - (src[:, j] - src[:, i]).norm(dim = -1) for i, j in BONE_EDGES]))

# ----- 7. Extract Keypoints -----
def extract_keypoints(batch, fa):
    kps = []
    for img in batch:
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        lms = fa.get_landmarks(arr)
        if not lms: 
            raise RuntimeError("No face found")
        kps.append(torch.tensor(lms[0], dtype = torch.float32))
    return torch.stack(kps).to(batch.device)

# ----- 8. Training -----
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default= '/root/video_generation/video_enhancement/GFPGAN/pure_talking_faces')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save_dir', type=str, default='keypoints_checkpoint')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok = True)

    train_ds = SingleFrameDataset(args.root_dir, True)
    val_ds = SingleFrameDataset(args.root_dir, False)

    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = True)
    val_loader = DataLoader(val_ds, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = True)
    
    fa = FaceAlignment(LandmarksType.THREE_D, flip_input = False)
    model = CanonicalPointNet().to(device)

    opt = optim.Adam(model.parameters(), lr = args.lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode = 'min', factor = 0.5, patience = 5)

    mse = nn.MSELoss()
    best_val = float('inf')

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss=0
        for img in tqdm(train_loader, desc=f"Train {epoch}"):
            img = img.to(device)

            try: 
                kp = extract_keypoints(img, fa)
            except RuntimeError: 
                continue

            can = model(kp)
            rec = rigid_procrustes_alignment(kp, can)
            rec_loss = mse(rec, kp)

            B = kp.size(0)
            R = random_rotation_matrix(B, device)
            mu = kp.mean(1, True)
            kp_r = (kp - mu) @ R.transpose(1, 2) + mu
            can_r = model(kp_r)
            mu_c = can.mean(1, True)
            can_gt = (can - mu_c) @ R.transpose(1,2) + mu_c
            eq_loss = mse(can_r, can_gt)

            bone_loss = bone_loss_fn(kp, can)
            loss = rec_loss + 0.3 * eq_loss + 0.1 * bone_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        val_count = 0

        with torch.no_grad():
            for img in tqdm(val_loader, desc = f"Val {epoch}"):
                img = img.to(device)
                try: 
                    kp = extract_keypoints(img, fa)
                except RuntimeError: 
                    continue
                can = model(kp); 
                rec = rigid_procrustes_alignment(kp, can)
                val_loss += mse(rec, kp).item()
                val_count += 1

        avg_tr = total_loss / len(train_loader)
        avg_vl = val_loss / (val_count or 1)

        sched.step(avg_vl)
        print(f"Epoch {epoch} — tr {avg_tr:.4f}, vl_rec {avg_vl:.4f}")

        if avg_vl < best_val:
            best_val = avg_vl
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pth'))

    torch.save(model.state_dict(),os.path.join(args.save_dir, 'final.pth'))
    print("Training complete.")
