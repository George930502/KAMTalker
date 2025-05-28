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

@torch.no_grad()
def eval(args):
    # Load and prepare models
    g_models = {
        "AFE": AppearanceFeatureExtraction(),
        "CKD": CanonicalKeypointDetector(num_kp=40),
        "HPE_EDE": HeadPoseEstimator_ExpressionDeformationEstimator(num_kp=40),
        "MFE": MotionFieldEstimator(num_kp=40),
        "Generator": Generator()
    } 
    ckp_path = os.path.join(args.ckp_dir, f"{str(args.ckp_epoch).zfill(8)}-iter{args.ckp_iter:06d}-checkpoint.pth.tar")
    checkpoint = torch.load(ckp_path, map_location=torch.device("cuda"), weights_only=True)
    for k, v in g_models.items():
        v.cuda()
        v.load_state_dict(checkpoint[k])
        v.eval()

    output_frames = []
    
    # Load source image
    s = img_as_float32(io.imread(args.source))[:, :, :3]
    s = np.array(s, dtype="float32").transpose((2, 0, 1))
    s = torch.from_numpy(s).cuda().unsqueeze(0)
    s = F.interpolate(s, size=(256, 256))
    s_np = s.squeeze(0).data.cpu().numpy().transpose(1, 2, 0)  # HWC for later concatenation
    s_np = (255 * s_np.clip(0, 1)).astype(np.uint8)  # Scale to [0, 255]

    fs = g_models["AFE"](s)
    kp_c = g_models["CKD"](s)
    yaw, pitch, roll, t, delta = g_models["HPE_EDE"](s)
    kp_s, Rs = transformkp(yaw, pitch, roll, kp_c, t, delta)

    # Read driving video/frames
    reader = get_reader(args.driving)
    video_frames = []
    for frame in reader:
        frame = img_as_float32(frame)
        if frame.shape[-1] == 4:  # Convert RGBA to RGB if necessary
            frame = frame[..., :3]
        if len(frame.shape) == 2:  # Convert grayscale to RGB
            frame = gray2rgb(frame)
        video_frames.append(frame)
    video_array = np.array(video_frames)
    
    # Process each frame in the driving video
    for img in video_array:
        img = np.array(img, dtype="float32").transpose((2, 0, 1))
        img = torch.from_numpy(img).cuda().unsqueeze(0)
        img = F.interpolate(img, size=(256, 256))

        yaw, pitch, roll, t, delta = g_models["HPE_EDE"](img)
        kp_d, Rd = transformkp(yaw, pitch, roll, kp_c, t, delta)
        deformation, occlusion = g_models["MFE"](fs, kp_s, kp_d, Rs, Rd)
        generated_d = g_models["Generator"](fs, deformation, occlusion)
    
        generated_d = generated_d.squeeze(0).data.cpu().numpy()
        generated_d = np.transpose(generated_d, [1, 2, 0])
        generated_d = generated_d.clip(0, 1)
        generated_d = (255 * generated_d).astype(np.uint8)
        output_frames.append(generated_d)
        
    # Save the generated output as a GIF (you can change extension if desired)
    imageio.mimsave(args.output, output_frames)
    print(f"Result saved in {args.output}")

def main_iterate(args):
    """
    Iterate over every subfolder inside the input root.
    Each subfolder is assumed to have a 'source.jpg' and a driving file 
    ('driving.mp4' or 'driving.jpg'). The output (generated video) is saved into 
    the same subfolder as 'output.gif'.
    """
    base_dir = args.input_root
    subfolders = sorted(os.listdir(base_dir))
    for folder in subfolders:
        subfolder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(subfolder_path):
            continue

        # Construct source and driving paths.
        source_path = os.path.join(subfolder_path, "source.jpg")
        # Prefer driving.mp4 if it exists; otherwise, use driving.jpg.
        if os.path.exists(os.path.join(subfolder_path, "driving.mp4")):
            driving_path = os.path.join(subfolder_path, "driving.mp4")
        else:
            driving_path = os.path.join(subfolder_path, "driving.jpg")
        
        # Set output file path
        if os.path.exists(os.path.join(subfolder_path, "driving.mp4")):
            output_path = os.path.join(subfolder_path, "output.mp4")
        else:
            output_path = os.path.join(subfolder_path, "output.jpg")
        
        # Create a new namespace for this iteration.
        new_args = argparse.Namespace(
            ckp_dir=args.ckp_dir,
            output=output_path,
            ckp_epoch=args.ckp_epoch,
            ckp_iter=args.ckp_iter,
            source=source_path,
            driving=driving_path
        )
        
        print(f"\nProcessing subfolder: {subfolder_path}")
        eval(new_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face-vid2vid")
    
    def str2bool(s):
        return s.lower().startswith("t")
    
    parser.add_argument("--ckp_dir", type=str, default="record/ckp(40)", help="Checkpoint directory")
    parser.add_argument("--ckp_epoch", type=int, default=7, help="Checkpoint epoch")
    parser.add_argument("--ckp_iter", type=int, default=28125, help="Checkpoint iteration")
    # These are default file paths if not iterating over subfolders.
    #parser.add_argument("--source", type=str, default="results/1/test2/source.jpg", help="Path to source image")
    #parser.add_argument("--driving", type=str, default="results/1/test2/driving.jpg", help="Path to driving video or image")
    #parser.add_argument("--output", type=str, default="results/1/output.gif", help="Output file path for generated video")
    # If set to True, the code will iterate through all subfolders in the input root.
    parser.add_argument("--iterate", type=str2bool, default=True, help="Whether to iterate over all subfolders in the input root")
    # The input root directory that contains all subfolders with source and driving inputs.
    parser.add_argument("--input_root", type=str, default="results/1", help="Input root directory containing subfolders")
    
    args = parser.parse_args()
    
    if args.iterate:
        main_iterate(args)
    else:
        eval(args)
