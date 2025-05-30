import argparse
import numpy as np
import torch, torchvision
import torch.nn.functional as F
import imageio
import os
from skimage import io
from skimage.util import img_as_float32
from imageio import get_reader
from skimage.color import gray2rgb
from modules.utils import *
from modules.Eapp import AppearanceFeatureExtraction
from modules.MotionFieldEstimator import MotionFieldEstimator
from modules.Generator import Generator
from modules.Emtn_gt import Hopenet
from face_alignment import FaceAlignment, LandmarksType
import cv2


def extract_canonical_kp(image_tensor, fa_ckd):
    B = image_tensor.shape[0]
    keypoints_list = []
    for i in range(B):
        img_tensor = image_tensor[i]  # (3, H, W)
        image_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255
        image_np = image_np.astype(np.uint8)
        landmarks = fa_ckd.get_landmarks(image_np)
        if landmarks is None or len(landmarks) == 0:
            return None
        keypoints_list.append(torch.tensor(landmarks[0], dtype=torch.float32))
    keypoints_batch = torch.stack(keypoints_list, dim=0).cuda()  # shape (B, 68, 3)
    return keypoints_batch


@torch.no_grad()
def eval(args):
    g_models = {
        "AFE": AppearanceFeatureExtraction(),
        "MFE": MotionFieldEstimator(args.num_kp),
        "Generator": Generator()
    } 
    ckp_path = os.path.join(args.ckp_dir, f"{str(args.ckp_epoch).zfill(8)}-iter{args.ckp_iter:06d}-checkpoint.pth.tar")
    checkpoint = torch.load(ckp_path, map_location=torch.device("cuda"), weights_only=True)

    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()
    
    fa = FaceAlignment(LandmarksType.THREE_D, flip_input=False, device="cuda")

    s = img_as_float32(io.imread(args.source))[:, :, :3]   # (256, 256, 3) => (H, W, C)
    s = s.transpose((2, 0, 1))  # (C, H, W)
    s = torch.from_numpy(s).cuda().unsqueeze(0)   # (1, C, H, W)

    s_np = s.squeeze(0).data.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC for concatenation
    s_np = (255 * s_np.clip(0, 1)).astype(np.uint8)  # Scale to [0, 255]

    pretrained_HPNet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66).cuda()
    pretrained_HPNet.load_state_dict(torch.load("pretrained/hopenet_robust_alpha1.pkl", map_location = torch.device("cuda"), weights_only = True))

    with torch.no_grad():
        pretrained_HPNet.eval()
    
    yaw_s, pitch_s, roll_s = pretrained_HPNet(F.interpolate(apply_imagenet_normalization(s), size = (224, 224)))
    
    fs = g_models["AFE"](s)
    kp_s = extract_canonical_kp(s, fa)    
    Rs = getRotationMatrix(yaw_s, pitch_s, roll_s)

    cap = cv2.VideoCapture(args.driving)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_array = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = img_as_float32(frame)
        video_array.append(frame)
    cap.release()

    output_frames = []

    for img in video_array:
        img = np.array(img, dtype="float32").transpose((2, 0, 1))
        img = torch.from_numpy(img).cuda().unsqueeze(0)

        kp_d = extract_canonical_kp(img, fa)    
        yaw_d, pitch_d, roll_d = pretrained_HPNet(F.interpolate(apply_imagenet_normalization(img), size = (224, 224)))
        Rd = getRotationMatrix(yaw_d, pitch_d, roll_d)
        deformation, occlusion = g_models["MFE"](fs, kp_s, kp_d, Rs, Rd)
        generated_d = g_models["Generator"](fs, deformation, occlusion)
    
        generated_d = torch.cat((img, generated_d), dim=3)
        generated_d = generated_d.squeeze(0).data.cpu().numpy()
        generated_d = np.transpose(generated_d, [1, 2, 0])
        generated_d = generated_d.clip(0, 1)
        generated_d = (255 * generated_d).astype(np.uint8)

        combined_frame = np.concatenate((s_np, generated_d), axis=1)  # Concatenate horizontally  
        output_frames.append(combined_frame)
        
    with imageio.get_writer(args.output, fps=fps) as writer:
        for frame in output_frames:
            writer.append_data(frame)

    print(f"Result in {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face-vid2vid")
    parser.add_argument("--ckp_dir", type=str, default="record/ckp", help="Checkpoint dir")
    parser.add_argument("--output", type=str, default="demo/output.mp4", help="Output video")
    parser.add_argument("--ckp_epoch", type=int, default=10, help="Checkpoint epoch")
    parser.add_argument("--ckp_iter", type=int, default=25000, help="Checkpoint iteration")
    parser.add_argument("--source", type=str, default="demo/source.jpg", help="Source image, f for face frontalization, r for reconstruction")
    parser.add_argument("--driving", type=str, default="demo/driving.mp4", help="Driving dir")
    parser.add_argument("--num_kp", type=int, default=68, help="Number of keypoints")

    args = parser.parse_args()
    eval(args)