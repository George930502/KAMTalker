import os
import sys
import argparse
import numpy as np
import torch
from torchvision import transforms
from imageio import get_reader
from skimage.util import img_as_float32
from skimage.color import gray2rgb
from face_alignment import FaceAlignment, LandmarksType

# Add project root for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Stage1.modules.utils import getRotationMatrix
from Stage1.modules.Emtn import HeadPoseEstimator_ExpressionDeformationEstimator


def read_video(path):
    """
    Read video file (.mp4 or .gif) and return array of frames (T, H, W, 3) in float32 [0,1].
    """
    if not path.lower().endswith(('.mp4', '.gif')):
        raise ValueError(f"Unsupported video format: {path}")
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


def extract_keypoints(batch: torch.Tensor, fa: FaceAlignment) -> torch.Tensor:
    """
    Extract 3D facial keypoints from a batch tensor of shape (B, 3, H, W).
    Returns (B, K, 3) on same device.
    """
    device = batch.device
    kps = []
    for img in batch:
        arr = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        lms = fa.get_landmarks(arr)
        if not lms:
            raise RuntimeError("No face detected in frame.")
        kps.append(torch.tensor(lms[0], dtype=torch.float32, device=device))
    return torch.stack(kps, dim=0)


def process_video(path: str,
                  hpe_model: HeadPoseEstimator_ExpressionDeformationEstimator,
                  fa: FaceAlignment,
                  transform,
                  device: torch.device) -> dict:
    """
    Process a single video to extract rotation matrices, translation vectors, deformation matrices and keypoints for each frame.
    Returns dict of numpy arrays.
    """
    frames = read_video(path)
    if frames.shape[0] < 2:
        print(f"Skipping {path}: fewer than 2 frames.")
        return {}

    tensor_frames = [transform(frame).to(device) for frame in frames]

    rot_list, trans_list, deform_list, kps_list = [], [], [], []

    hpe_model.eval()
    with torch.no_grad():
        for frame_tensor in tensor_frames:
            inp = frame_tensor.unsqueeze(0)
            yaw, pitch, roll, translation, deformation = hpe_model(inp)
            # compute rotation matrix
            rotation = getRotationMatrix(yaw, pitch, roll)  # (1,3,3)

            # extract keypoints
            kp = extract_keypoints(inp, fa)

            rot_list.append(rotation.cpu().numpy().astype(np.float16))
            trans_list.append(translation.cpu().numpy().astype(np.float16))
            deform_list.append(deformation.cpu().numpy().astype(np.float16))
            kps_list.append(kp.cpu().numpy().astype(np.float16))

    return {
        'rotation': np.vstack(rot_list),            # (T,3,3)
        'translation': np.vstack(trans_list),  # (T,3)
        'deformation': np.vstack(deform_list) # (T,K,3)
        #'keypoints': np.vstack(kps_list)       # (T,K,3)
    }


def get_split_files(base_dir: str):
    """
    Return dict with splits ['train','val','test'] mapping to sorted list of .mp4 or .gif files.
    """
    splits = {}
    for split in ['train','val','test']:
        folder = os.path.join(base_dir, split)
        if not os.path.isdir(folder):
            print(f"Warning: {folder} not found.")
            splits[split] = []
        else:
            splits[split] = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith(('.mp4', '.gif'))
            ])
    return splits


def main():
    parser = argparse.ArgumentParser(description="Extract GT: rotation, translation, deformation, keypoints.")
    parser.add_argument('--video_base', required=True, help="Root dir with train/val/test subfolders.")
    parser.add_argument('--gt_base', required=True, help="Output dir for .npz files.")
    parser.add_argument('--ckpt', required=True, help="Path to pretrained HPE model checkpoint.")
    args = parser.parse_args()

    os.makedirs(args.gt_base, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fa = FaceAlignment(LandmarksType.THREE_D, flip_input=False, device="cuda")
    hpe_model = HeadPoseEstimator_ExpressionDeformationEstimator(num_kp=68).to(device)
    hpe_model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True), strict=False)
    hpe_model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    split_map = get_split_files(args.video_base)

    for split, files in split_map.items():
        print(f"Processing split '{split}': {len(files)} videos.")
        out_dir = os.path.join(args.gt_base, split)
        os.makedirs(out_dir, exist_ok=True)

        for vid in files:
            vid_path = os.path.join(args.video_base, split, vid)
            name = os.path.splitext(vid)[0]
            save_path = os.path.join(out_dir, f"{name}_gt.npz")
            if os.path.exists(save_path):
                print(f"Skipping existing: {save_path}")
                continue

            data = process_video(vid_path, hpe_model, fa, transform, device)
            if not data:
                continue
            np.savez_compressed(save_path, **data)
            print(f"Saved: {save_path}")

if __name__ == '__main__':
    main()