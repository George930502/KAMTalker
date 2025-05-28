import argparse
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import os
from skimage import io
from skimage.util import img_as_float32
from imageio import get_reader
from skimage.color import gray2rgb
from modules.utils import *
from modules.Eapp import AppearanceFeatureExtraction
from modules.Emtn import HeadPoseEstimator_ExpressionDeformationEstimator
from modules.KPDetector import CanonicalKeypointDetector
from modules.MotionFieldEstimator import MotionFieldEstimator
from modules.Generator import Generator
from modules.Discriminator import Discriminator
import cv2


@torch.no_grad()
def eval(args):
    g_models = {
        "AFE": AppearanceFeatureExtraction(),
        "CKD": CanonicalKeypointDetector(args.num_kp),
        "HPE_EDE": HeadPoseEstimator_ExpressionDeformationEstimator(args.num_kp),
        "MFE": MotionFieldEstimator(args.num_kp),
        "Generator": Generator()
    } 
    ckp_path = os.path.join(args.ckp_dir, f"{str(args.ckp_epoch).zfill(8)}-iter{args.ckp_iter:06d}-checkpoint.pth.tar")
    checkpoint = torch.load(ckp_path, map_location=torch.device("cuda"), weights_only=True)
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()
    
    s = img_as_float32(io.imread(args.source))[:, :, :3]
    s = np.array(s, dtype="float32").transpose((2, 0, 1))
    s = torch.from_numpy(s).cuda().unsqueeze(0)
    s = F.interpolate(s, size=(256, 256))
    s_np = s.squeeze(0).data.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC for concatenation
    s_np = (255 * s_np.clip(0, 1)).astype(np.uint8)  # Scale to [0, 255]

    fs = g_models["AFE"](s)
    kp_c = g_models["CKD"](s)
    yaw, pitch, roll, t, delta = g_models["HPE_EDE"](s)
    kp_s, Rs = transformkp(yaw, pitch, roll, kp_c, t, delta)

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
        img = F.interpolate(img, size=(256, 256))

        yaw, pitch, roll, t, delta = g_models["HPE_EDE"](img)
        kp_d, Rd = transformkp(yaw, pitch, roll, kp_c, t, delta)
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

    def str2bool(s):
        return s.lower().startswith("t")

    parser.add_argument("--ckp_dir", type=str, default="record/ckp1", help="Checkpoint dir")
    parser.add_argument("--output", type=str, default="demo/output.mp4", help="Output video")
    parser.add_argument("--ckp_epoch", type=int, default=27, help="Checkpoint epoch")
    parser.add_argument("--ckp_iter", type=int, default=28125, help="Checkpoint iteration")
    parser.add_argument("--source", type=str, default="demo/source.jpg", help="Source image, f for face frontalization, r for reconstruction")
    parser.add_argument("--driving", type=str, default="demo/driving.mp4", help="Driving dir")
    parser.add_argument("--num_kp", type=int, default=20, help="Number of keypoints")


    args = parser.parse_args()
    eval(args)