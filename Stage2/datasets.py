import sys
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path

class VideoAudioDataset(Dataset):
    def __init__(
        self,
        gt_dir: str,
        audio_dir: str,
        split: str,
        n_motions: int = 50,
        audio_sampling_rate: int = 16000,
        video_fps: float = 25.0,
        crop_strategy: str = 'random',
    ):
        assert split in ('train', 'val', 'test'), "split must be 'train','val', or 'test'"
        self.gt_dir = Path(gt_dir) / split
        self.audio_dir = Path(audio_dir) / split
        
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Ground-truth directory not found: {self.gt_dir}")
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        # entries from ground-truth files (without suffix)
        self.entries = [p.stem.replace('_gt', '')
                        for p in sorted(self.gt_dir.glob('*_gt.npz'))]

        self.n_motions = n_motions
        self.audio_sampling_rate = audio_sampling_rate
        self.video_fps = video_fps
        self.audio_clip_samples = int(round(audio_sampling_rate * n_motions / video_fps))
        assert crop_strategy in ('begin', 'random', 'end'), f"Unknown crop_strategy: {crop_strategy}"
        self.crop_strategy = crop_strategy

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        base = self.entries[idx]
        gt_path = self.gt_dir / f"{base}_gt.npz"
        audio_path = self.audio_dir / f"{base}_audio.aac"
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file missing: {gt_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file missing: {audio_path}")

        # load ground-truth npz
        data = np.load(gt_path)
        rot_seq = data['rotation']          # (T,3,3)
        trans_seq = data['translation']  # (T,3)
        deform_seq = data['deformation'] # (T,K,3)

        T = rot_seq.shape[0]
        if T < self.n_motions:
            raise ValueError(f"Not enough frames: required {self.n_motions}, got {T}")
        if self.crop_strategy == 'begin':
            start = 0
        elif self.crop_strategy == 'random':
            start = random.randint(0, T - self.n_motions)
        else:
            start = T - self.n_motions

        # build driving features
        feats = []
        for i in range(start, start + self.n_motions):
            rot = rot_seq[i]                   # (3,3)
            trans = trans_seq[i]           # (3,)
            deform = deform_seq[i].reshape(-1)  # (K*3,)
            vec = np.concatenate([rot.reshape(-1), trans, deform], axis=0)
            feats.append(vec)
        driving_features = torch.from_numpy(np.stack(feats, 0)).float()  # (n_motions, D)

        # source feature from first frame
        rot0 = rot_seq[0]
        trans0 = trans_seq[0]
        deform0 = deform_seq[0].reshape(-1)
        source_vec = np.concatenate([rot0.reshape(-1), trans0, deform0], axis=0)
        source_feature = torch.from_numpy(source_vec).float()

        # load audio clip aligned with driving
        wav, sr = torchaudio.load(str(audio_path))
        if sr != self.audio_sampling_rate:
            raise ValueError(f"Sampling rate mismatch: expected {self.audio_sampling_rate}, got {sr}")
        wav = wav[0]  # mono
        total_samples = wav.shape[0]
        start_sample = int(round(start / self.video_fps * self.audio_sampling_rate))
        end_sample = start_sample + self.audio_clip_samples
        if end_sample <= total_samples:
            clip = wav[start_sample:end_sample]
        else:
            pad_len = end_sample - total_samples
            clip = torch.cat([wav[start_sample:], torch.zeros(pad_len)])

        return clip, driving_features, source_feature


# Example usage:
if __name__ == "__main__":
    gt_directory = "/root/video_generation/Stage2/ground_truth"
    audio_directory = "/root/video_generation/video_enhancement/GFPGAN/extracted_audio"
    dataset = VideoAudioDataset(
        gt_dir=gt_directory,
        audio_dir=audio_directory,
        split='train',
        n_motions=50,
        audio_sampling_rate=16000,
        video_fps=25.0,
        crop_strategy='begin'
    )
    clip, driving_features, source_feature = dataset[0]
    print("Audio shape:", clip.shape)
    print("Driving features shape:", driving_features.shape)
    print("Source feature shape:", source_feature.shape)
