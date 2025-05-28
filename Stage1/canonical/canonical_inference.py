import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import torchvision.transforms as transforms
from face_alignment import FaceAlignment, LandmarksType

# ----- Helper: Load Image -----
def load_image(path, size=None):
    img = Image.open(path).convert('RGB')
    tfm = []
    if size:
        tfm.append(transforms.Resize(size))
    tfm.append(transforms.ToTensor())
    return transforms.Compose(tfm)(img).unsqueeze(0)

# ----- MLP Residual Block (same as training) -----
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

# ----- CanonicalPointNet (matches training) -----
class CanonicalPointNet(nn.Module):
    def __init__(self, point_dim=3, emb_dim=64, global_dim=256):
        super().__init__()
        # initial embedding
        self.init_embed = nn.Sequential(
            nn.Linear(point_dim, emb_dim), nn.ReLU(),
            ResMLP(emb_dim), ResMLP(emb_dim)
        )
        # global feature extractor with skip connection
        self.global_net = nn.Sequential(
            nn.Linear(emb_dim, global_dim), nn.ReLU(),
            nn.Linear(global_dim, global_dim)
        )
        self.global_skip = nn.Linear(emb_dim, global_dim)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim + global_dim, global_dim), nn.ReLU(),
            nn.Linear(global_dim, point_dim)
        )

    def forward(self, kp):
        # kp: (B, N, 3)
        B, N, _ = kp.shape
        # embed each point
        f = self.init_embed(kp)                  # (B, N, emb_dim)
        # global pooling
        g, _ = f.max(dim=1)                      # (B, emb_dim)
        # global features + skip
        g_feat = self.global_net(g) + self.global_skip(g)  # (B, global_dim)
        # broadcast
        g_exp = g_feat.unsqueeze(1).expand(-1, N, -1)       # (B, N, global_dim)
        # concatenate and decode
        feat = torch.cat([f, g_exp], dim=2)   # (B, N, emb_dim+global_dim)
        return self.decoder(feat)

# ----- Extract 3D Keypoints -----
def extract_keypoints(img_batch, fa):
    kps = []
    for img in img_batch:
        arr = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        lms = fa.get_landmarks(arr)
        if not lms:
            continue
        kps.append(torch.tensor(lms[0], dtype=torch.float32))
    if not kps:
        raise RuntimeError("No face detected.")
    return torch.stack(kps, dim=0)

# ----- Inference & Visualization -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',        type=str, required=True, help='Input image')
    parser.add_argument('--model_path', type=str, required=True, help='Trained model .pth')
    parser.add_argument('--resize',     type=int, nargs=2, default=[256,256])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    model = CanonicalPointNet().to(device)
    ckpt  = torch.load(args.model_path, weights_only=True, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    # face alignment
    fa = FaceAlignment(LandmarksType.THREE_D, flip_input=False)
    # read and process image
    img = load_image(args.img, size=tuple(args.resize)).to(device)
        # extract GT keypoints (B,N,3) for batch
    kp_gt = extract_keypoints(img, fa).to(device)  # shape: (batch_size, N, 3)
    # if single image, batch_size==1, so shape is (1,N,3)

    # predict canonical
    with torch.no_grad():
        kp_pred = model(kp_gt)               # (1,N,3)
    # convert to numpy
    kp_gt_np   = kp_gt.squeeze(0).cpu().numpy()    # (N,3)
    kp_pred_np = kp_pred.squeeze(0).cpu().numpy()  # (N,3)
    # plot
    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(kp_gt_np[:,0], kp_gt_np[:,1], kp_gt_np[:,2], c='blue', label='GT Keypoints')
    ax.scatter(kp_pred_np[:,0], kp_pred_np[:,1], kp_pred_np[:,2], c='red', s=20, label='Canonical Keypoints')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); plt.tight_layout(); plt.savefig('compare.png')

if __name__ == '__main__':
    main()
