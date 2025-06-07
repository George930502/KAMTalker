import argparse
import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchaudio

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from video_generation.Stage1_origin.logger import Logger
from model import DiffTalkingHead


def pad_to_len(t: torch.Tensor, length: int, dim: int, mode: str = 'zero') -> torch.Tensor:
    """
    Pad tensor t along dimension dim to target length.
    mode: 'zero' pads with zeros; 'replicate' pads with last value.
    """
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


def load_stage1_model(ckp_dir, vis_dir, epoch, iteration, device):
    logger = Logger(num_kp = 20, ckp_dir=ckp_dir, vis_dir=vis_dir, dataloader=None, lr=0)
    logger.load_cpk(epoch=epoch, iteration=iteration)
    g_full = logger.g_full.to(device).eval()
    return g_full


def load_stage2_model(cfg_path, ckpt_path, device):
    with open(cfg_path, 'r') as f:
        cfg = argparse.Namespace(**json.load(f))
    model = DiffTalkingHead(cfg, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Inference for talking head generation with chunked sampling and source prepend")
    parser.add_argument('--source_img', type=str, required=True, help='Path to source image (256x256 RGB)')  # /root/video_generation/Stage2/inference/img/source.jpg
    parser.add_argument('--audio_path', type=str, required=True, help='Path to input audio file')  # /root/video_generation/Stage2/inference/audio/0eEnKKS0mXU-00026_audio.aac
    parser.add_argument('--stage1_ckp_dir', type=str, default='/root/video_generation/Stage1/record/ckp1')
    parser.add_argument('--stage1_vis_dir', type=str, default='/root/video_generation/Stage1/record/vis1')
    parser.add_argument('--stage1_epoch', type=int, default=27)
    parser.add_argument('--stage1_iter', type=int, default=28125)
    parser.add_argument('--stage2_cfg', type=str, default='/root/video_generation/Stage2/experiments/run_250509_210239/hyperparameters.json')
    parser.add_argument('--stage2_ckpt', type=str, default='/root/video_generation/Stage2/experiments/run_250509_210239/checkpoints/best.pt')
    parser.add_argument('--output_dir', type=str, default='/root/video_generation/Stage2/inference/generated')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--audio_sr', type=int, default=16000)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--pad_mode', type=str, choices=['zero','replicate'], default='zero')
    parser.add_argument('--use_indicator', type=bool, default=True)
    parser.add_argument('--dynamic_threshold_ratio', type=float, default=0.9)
    parser.add_argument('--dynamic_threshold_min', type=float, default=1.0)
    parser.add_argument('--dynamic_threshold_max', type=float, default=4.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dynamic Thresholding
    if args.dynamic_threshold_ratio > 0:
        dynamic_threshold = (args.dynamic_threshold_ratio, args.dynamic_threshold_min, args.dynamic_threshold_max)
    else:
        dynamic_threshold = None

    # Load models
    g_full = load_stage1_model(args.stage1_ckp_dir, args.stage1_vis_dir, args.stage1_epoch, args.stage1_iter, device)
    diffusion_model = load_stage2_model(args.stage2_cfg, args.stage2_ckpt, device)

    # Transforms
    img_transform = transforms.Compose([transforms.ToTensor()])
    to_pil = transforms.ToPILImage()

    # Save source frame as first frame
    src_img = Image.open(args.source_img).convert('RGB')
    src_tensor = img_transform(src_img).unsqueeze(0).to(device)
    first_frame = os.path.join(args.output_dir, 'frame_0000.png')
    src_img.save(first_frame)
    print(f"Saved source frame -> {first_frame}")

    # Extract source features
    with torch.no_grad():
        fs = g_full.AFE(src_tensor)
        kp_c = g_full.CKD(src_tensor)

    # Load audio
    wav, sr = torchaudio.load(args.audio_path)
    wav = wav.mean(dim=0, keepdim=True) if wav.ndim>1 else wav.unsqueeze(0)
    if sr != args.audio_sr:
        wav = torchaudio.functional.resample(wav, sr, args.audio_sr)
    audio = wav.to(device)

    # Compute lengths
    total_samples = audio.shape[-1]
    audio_unit = int(args.audio_sr / args.fps)  # 640
    clip_len = int(total_samples / args.audio_sr * args.fps)
    stride = diffusion_model.n_motions
    n_subdiv = 1 if clip_len <= stride else math.ceil(clip_len / stride)

    n_audio_samples = round(audio_unit * diffusion_model.n_motions)
    n_padding_audio_samples = n_audio_samples * n_subdiv - total_samples
    n_padding_frames = math.ceil(n_padding_audio_samples / audio_unit)

    pad_mode = args.pad_mode

    if n_padding_audio_samples > 0:
        if pad_mode == 'zero':
            padding_value = 0
        elif pad_mode == 'replicate':
            padding_value = audio[-1]
        else:
            raise ValueError(f'Unknown pad mode: {pad_mode}')
        audio = F.pad(audio, (0, n_padding_audio_samples), value=padding_value)

    audio_feat = diffusion_model.extract_audio_feature(audio, diffusion_model.n_motions * n_subdiv)
    
    # Chunk sampling
    chunks = []
    prev_motion = None
    prev_audio_feat = None

    for i in range(n_subdiv):
        start_f = i * stride
        end_f = start_f + diffusion_model.n_motions
        if args.use_indicator:
            indicator = torch.ones((1, diffusion_model.n_motions), device=device, dtype=torch.bool)
            if i == n_subdiv - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0
        else:
            indicator = None

        # s = start_f * audio_unit
        # e = end_f * audio_unit
        # chunk = audio[:, s:e]

        audio_in = audio_feat[:, start_f:end_f].expand(1, -1, -1)

        with torch.no_grad():
            if i == 0:
                motion, noise, prev_audio_feat = diffusion_model.sample(
                    audio_or_feat=audio_in, 
                    indicator=indicator, 
                    flexibility=0.5,
                    dynamic_threshold=dynamic_threshold)
            else:
                motion, noise, prev_audio_feat = diffusion_model.sample(
                    audio_or_feat=audio_in,
                    prev_motion_feat=prev_motion,
                    prev_audio_feat=prev_audio_feat,
                    motion_at_T=noise,
                    indicator=indicator,
                    flexibility=0.5,
                    dynamic_threshold=dynamic_threshold
                )

        prev_motion = motion[:, -diffusion_model.n_prev_motions:].clone()
        prev_audio_feat = prev_audio_feat[:, -diffusion_model.n_prev_motions:]

        if i == n_subdiv - 1 and n_padding_frames > 0:
            motion = motion[:, :-n_padding_frames]  # delete padded frames

        # if i == 0:        
        #    chunks.append(motion)
        # else:
        #    chunks.append(motion[:, diffusion_model.n_prev_motions:])

        chunks.append(motion)

    # Concatenate and trim
    # motions = torch.cat(chunks, dim=1)[0][:clip_len]
    motions = torch.cat(chunks, dim=1)
    #print(motions.shape)   # torch.Size([1, 134, 72])

    # Generate driven frames
    Rs = torch.eye(3, device=device).unsqueeze(0)
    for idx in range(motions.shape[1]):
        latent = motions[0, idx] 
        rot = latent[:9].view(1,3,3)
        trans = latent[9:12].view(1,3)
        deform = latent[12:].view(1,20,3)
        kp_d = torch.matmul(rot.unsqueeze(1), kp_c.unsqueeze(-1)).squeeze(-1) + trans.unsqueeze(1) + deform
        with torch.no_grad():
            flow, mask = g_full.MFE(fs, kp_c, kp_d, Rs, rot)
            gen = g_full.Generator(fs, flow, mask)
        img = to_pil(gen.squeeze(0).cpu())
        path = os.path.join(args.output_dir, f"frame_{idx+1:04d}.png")
        img.save(path)
        print(f"Saved frame {idx+1} -> {path}")

    # Combine to video with audio trim
    try:
        import subprocess
        out_vid = os.path.join(args.output_dir, 'output.mp4')
        # Construct ffmpeg command: frames + original audio
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(args.fps),
            '-i', os.path.join(args.output_dir, 'frame_%04d.png'),
            '-i', args.audio_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-shortest',
            out_vid
        ]
        subprocess.run(cmd, check=True)
        print(f"Saved final video with audio -> {out_vid}")
    except Exception as e:
        print(f"Failed to create video via ffmpeg: {e}")
        print("Ensure ffmpeg is installed and in PATH.")

        
if __name__ == '__main__':
    main()
