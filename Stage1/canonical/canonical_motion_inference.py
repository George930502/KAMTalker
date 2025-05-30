import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchvision.transforms as transforms
from face_alignment import FaceAlignment, LandmarksType

from imageio import get_reader
from skimage.util import img_as_float32
from skimage.color import gray2rgb

sys.path.append(os.path.abspath('..'))

from modules.Emtn import HeadPoseEstimator_ExpressionDeformationEstimator
from modules.utils import getRotationMatrix

# helper: load image
def load_image(path):
    img = Image.open(path).convert('RGB')
    tfm = []
    tfm.append(transforms.ToTensor())
    return transforms.Compose(tfm)(img).unsqueeze(0)

# helper: load video frames (with retry sampling)
def load_and_sample_frames(path, num_frames, random=True):
    reader = get_reader(path)
    frames_raw = []
    for frame in reader:
        img = img_as_float32(frame)
        if img.ndim == 2:
            img = gray2rgb(img)
        if img.shape[-1] == 4:
            img = img[..., :3]
        pil = Image.fromarray((img*255).astype('uint8'))
        frames_raw.append(transforms.ToTensor()(pil))
    T = len(frames_raw)
    if T == 0:
        return []
    if random or T < num_frames:
        idx = np.random.choice(T, num_frames, replace=(T < num_frames))
    else:
        idx = np.linspace(0, T-1, num_frames, dtype=int)
    return [frames_raw[i] for i in idx]

# extract 3D keypoints
def extract_keypoints(img_batch, fa):
    kps = []
    for img in img_batch:
        arr = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        lms = fa.get_landmarks(arr)
        if not lms:
            kps.append(None)
        else:
            kps.append(torch.tensor(lms[0], dtype=torch.float32))
    return kps

# inference: single image
def inference_image(model, img, fa, device):
    img = img.to(device)
    kps = extract_keypoints(img, fa)
    if kps[0] is None:
        # no face, skip
        return None
    kp = kps[0].to(device).unsqueeze(0)

    with torch.no_grad():
        yaw, pitch, roll, trans, deform = model(img)  # trans, deform shape: (1, 3)
        R = getRotationMatrix(yaw, pitch, roll)       # shape: (1, 3, 3), tensor
        # Do everything in torch first
        R_t = R.transpose(1, 2)                        # (1, 3, 3)
        canon = ((kp - trans.unsqueeze(1) - deform).bmm(R_t))[0].cpu().numpy()

    # Then convert for saving/return
    R_np = R[0].cpu().numpy()
    t = trans[0].cpu().numpy()
    d = deform[0].cpu().numpy()
    kp_np = kp[0].cpu().numpy()
        
    return R_np, t, d, kp_np, canon

# inference: multi-frame video
def inference_video(model, video_path, fa, device, num_frames):
    # try regular then random sampling if no face
    frames = load_and_sample_frames(video_path, num_frames, random=False)
    kps = extract_keypoints(frames, fa)
    valid = [i for i,pt in enumerate(kps) if pt is not None]
    if not valid:
        frames = load_and_sample_frames(video_path, num_frames, random=True)
        kps = extract_keypoints(frames, fa)
        valid = [i for i,pt in enumerate(kps) if pt is not None]
        if not valid:
            # give up
            return None
    imgs = torch.stack([frames[i] for i in valid], dim=0).to(device)
    kp_tensor = torch.stack([kps[i] for i in valid], dim=0).to(device)

    with torch.no_grad():
        yaw, pitch, roll, trans, deform = model(imgs)
        R_all = getRotationMatrix(yaw, pitch, roll)  # torch.Tensor
        canon = ((kp_tensor - trans.unsqueeze(1) - deform).bmm(R_all.transpose(1, 2)))

    R_all_np = R_all.cpu().numpy()
    t_all = trans.cpu().numpy()
    d_all = deform.cpu().numpy()
    canon_all = canon.cpu().numpy()
    kp_orig_all = kp_tensor.cpu().numpy()

    return R_all_np, t_all, d_all, kp_orig_all, canon_all

# visualization: single plot for video
def plot_video_keypoints(gt_all, canon_all, save_path):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gt_all[:,:,0].flatten(), gt_all[:,:,1].flatten(), gt_all[:,:,2].flatten(),
               c='blue', alpha=0.3, label='Original')
    ax.scatter(canon_all[:,:,0].flatten(), canon_all[:,:,1].flatten(), canon_all[:,:,2].flatten(),
               c='red', alpha=0.3, s=10, label='Canonical')
    ax.set_title('Video Inference: All Frames')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); plt.tight_layout(); plt.savefig(save_path)
    plt.close()

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['image','video'], required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='inference_out')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    NUM_KP = 68
    model = HeadPoseEstimator_ExpressionDeformationEstimator(num_kp=NUM_KP).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    fa = FaceAlignment(LandmarksType.THREE_D, flip_input=False)

    if args.mode == 'image':
        res = inference_image(model, load_image(args.input), fa, device)
        if res is not None:
            R, t, d, kp_o, kp_c = res
            np.save(os.path.join(args.save_dir, 'R.npy'), R)
            np.save(os.path.join(args.save_dir, 't.npy'), t)
            np.save(os.path.join(args.save_dir, 'd.npy'), d)
            plot_video_keypoints(kp_o[np.newaxis], kp_c[np.newaxis], os.path.join(args.save_dir, 'compare.png'))
        else:
            print('No face detected in image, skipping output')

    else:
        res = inference_video(model, args.input, fa, device, args.num_frames)
        if res is not None:
            R_all, t_all, d_all, kp_o_all, kp_c_all = res
            np.save(os.path.join(args.save_dir, 'R_all.npy'), R_all)
            np.save(os.path.join(args.save_dir, 't_all.npy'), t_all)
            np.save(os.path.join(args.save_dir, 'd_all.npy'), d_all)
            plot_video_keypoints(kp_o_all, kp_c_all, os.path.join(args.save_dir, 'compare_video.png'))
        else:
            print('No faces detected in any sampled frames, skipping output')

if __name__ == '__main__':
    main()
