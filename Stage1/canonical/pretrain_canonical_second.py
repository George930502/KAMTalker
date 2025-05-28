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
        items = os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, 'train')):
            self.train_dir = os.path.join(root_dir, 'train')
            self.test_dir  = os.path.join(root_dir, 'test')
        else:
            train_items, test_items = train_test_split(items, random_state=random_seed, test_size=test_size)
            self.train_dir = os.path.join(root_dir, 'train')
            self.test_dir  = os.path.join(root_dir, 'test')
            os.makedirs(self.train_dir, exist_ok=True)
            os.makedirs(self.test_dir,  exist_ok=True)
            for v in train_items:
                shutil.move(os.path.join(root_dir, v), self.train_dir)
            for v in test_items:
                shutil.move(os.path.join(root_dir, v), self.test_dir)
        self.is_train = is_train

    def _list(self):
        return [os.path.join(self.train_dir if self.is_train else self.test_dir, f)
                for f in os.listdir(self.train_dir if self.is_train else self.test_dir)]

    def __len__(self): return len(self._list())
    def __getitem__(self, idx):
        path = self._list()[idx]
        vid = read_video(path)
        idx_frame = np.random.randint(len(vid))
        img = vid[idx_frame].astype('float32').transpose(2,0,1)
        return img

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
            nn.Linear(point_dim, emb_dim), nn.ReLU(), ResMLP(emb_dim), ResMLP(emb_dim)
        )
        self.global_net  = nn.Sequential(
            nn.Linear(emb_dim, global_dim), nn.ReLU(), nn.Linear(global_dim, global_dim)
        )
        self.global_skip = nn.Linear(emb_dim, global_dim)
        self.decoder     = nn.Sequential(
            nn.Linear(emb_dim + global_dim, global_dim), nn.ReLU(), nn.Linear(global_dim, point_dim)
        )
    def forward(self, kp):
        B,N,_ = kp.shape
        f      = self.init_embed(kp)
        g,_    = f.max(dim=1)
        g_feat = self.global_net(g) + self.global_skip(g)
        g_exp  = g_feat.unsqueeze(1).expand(-1,N,-1)
        feat   = torch.cat([f, g_exp], dim=2)
        return self.decoder(feat)

# ----- 4. TransformNet: predict R, t, d -----
class TransformNet(nn.Module):
    def __init__(self, num_points, hidden_dim=512):
        super().__init__()
        in_dim  = num_points * 3
        out_dim = 9 + 3 + num_points*3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.num_points = num_points
    def forward(self, can):  # can: (B, N, 3)
        B,N,_ = can.shape
        x = can.view(B, -1)        # (B, N*3)
        p = self.net(x)            # (B, 9+3+N*3)
        R = p[:, :9].view(B,3,3)
        t = p[:, 9:12].view(B,1,3)
        d = p[:, 12:].view(B,N,3)
        return R, t, d

# ----- 5. Extract Keypoints -----
def extract_keypoints(batch, fa):
    kps=[]
    device=batch.device
    for img in batch:
        arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        lms=fa.get_landmarks(arr)
        if not lms: continue
        kps.append(torch.tensor(lms[0], dtype=torch.float32, device=device))
    if not kps: raise RuntimeError("No face found")
    return torch.stack(kps, dim=0)

# ----- 6. Training Pipeline -----
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, required=True)
    p.add_argument('--batch_size',type=int,default=16)
    p.add_argument('--epochs',    type=int,default=20)
    p.add_argument('--lr',        type=float,default=1e-4)
    p.add_argument('--save_dir',  type=str,default='keypoints_ckpt')
    args=p.parse_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    ds_train=SingleFrameDataset(args.root_dir,True)
    ds_val  =SingleFrameDataset(args.root_dir,False)
    ld_train=DataLoader(ds_train,batch_size=args.batch_size,shuffle=True,pin_memory=True)
    ld_val  =DataLoader(ds_val,  batch_size=args.batch_size,shuffle=False,pin_memory=True)
    fa     =FaceAlignment(LandmarksType.THREE_D,flip_input=False)
    model  =CanonicalPointNet().to(device)
    tnet   =TransformNet(num_points=68).to(device)
    opt    =optim.Adam(list(model.parameters())+list(tnet.parameters()),lr=args.lr)
    mse    =nn.MSELoss()
    best_val=float('inf')

    for epoch in range(1,args.epochs+1):
        model.train(); tnet.train(); total_loss=0
        for img in tqdm(ld_train,desc=f"Train {epoch}"):
            img=img.to(device)
            try: kp=extract_keypoints(img,fa)
            except: continue
            can = model(kp)
            R,t,d = tnet(can)
            rec=torch.bmm(can,R.transpose(1,2))+t+d
            loss=mse(rec,kp)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss+=loss.item()
        model.eval(); tnet.eval(); val_loss=0; val_n=0
        with torch.no_grad():
            for img in tqdm(ld_val,desc=f"Val {epoch}"):
                img=img.to(device)
                try: kp=extract_keypoints(img,fa)
                except: continue
                can=model(kp); R,t,d=tnet(can)
                rec=torch.bmm(can,R.transpose(1,2))+t+d
                val_loss+=mse(rec,kp).item(); val_n+=1
        avg_tr=total_loss/len(ld_train)
        avg_vl=val_loss/(val_n or 1)
        print(f"Epoch {epoch} — tr {avg_tr:.4f}, vl {avg_vl:.4f}")
        if avg_vl<best_val:
            best_val=avg_vl
            torch.save({'model':model.state_dict(),'tnet':tnet.state_dict()},os.path.join(args.save_dir,'best.pth'))
    torch.save({'model':model.state_dict(),'tnet':tnet.state_dict()},os.path.join(args.save_dir,'final.pth'))
    print("Training complete.")
