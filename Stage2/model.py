import torch
import torch.nn as nn
import torch.nn.functional as F

from common import PositionalEncoding, enc_dec_mask, pad_audio
# from transformers import HubertModel
from hubert import HubertModel

import argparse


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=0.02, s=0.008):
        super().__init__()

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, num_steps)
        elif mode == 'quadratic':
            betas = torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2
        elif mode == 'sigmoid':
            betas = torch.sigmoid(torch.linspace(-5, 5, num_steps)) * (beta_T - beta_1) + beta_1
        elif mode == 'cosine':
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alpha_bars = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f'Unknown diffusion schedule {mode}!')
        betas = torch.cat([torch.zeros(1), betas], dim=0)  # Padding: beta_0 = 0

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.shape[0]):
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.shape[0]):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.num_steps = num_steps
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = torch.randint(1, self.num_steps + 1, (batch_size,))
        return ts.tolist()

    def get_sigmas(self, t, flexibility=0):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class DiffTalkingHead(nn.Module):
    def __init__(self, args, device='cuda'):
        super().__init__()

        # Model parameters
        self.target = args.target         # e.g. 'noise' or 'sample'
        self.architecture = args.architecture

        # In this modified design, we define the motion feature as the concatenation of:
        # - Rotation matrix (3x3 = 9 values)
        # - Translation vector (3 values)
        # - Deformation matrix (68x3 = 204 values)
        # Total dimension = 9 + 3 + 204 = 216.
        self.motion_feat_dim = 9 + 3 + 204

        # Audio encoder
        self.audio_model = args.audio_model
        if self.audio_model == 'hubert':
            self.audio_encoder = HubertModel.from_pretrained('facebook/hubert-base-ls960', attn_implementation="eager")
            self.audio_encoder.feature_extractor._freeze_parameters()
            frozen_layers = [0, 1]
            for name, param in self.audio_encoder.named_parameters():
                if name.startswith("feature_projection"):
                    param.requires_grad = False
                if name.startswith("encoder.layers"):
                    layer = int(name.split(".")[2])
                    if layer in frozen_layers:
                        param.requires_grad = False
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')

        if args.architecture == 'decoder':
            self.audio_feature_map = nn.Linear(768, args.feature_dim)
        else:
            raise ValueError(f'Unknown architecture {args.architecture}!')

        self.fps = args.fps
        # In our case, we assume a single transformation prediction (n_motions can be set to 1)
        self.n_prev_motions = args.n_prev_motions
        self.n_motions = args.n_motions

        self.start_audio_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, args.feature_dim))
        self.start_motion_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, self.motion_feat_dim))

        # Diffusion model
        self.denoising_net = DenoisingNetwork(args, device)
        self.diffusion_sched = DiffusionSchedule(args.n_diff_steps, args.diff_schedule)

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, time_step=None, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_motion) ground-truth transformation parameters
            audio_or_feat: (N, L_audio) raw audio or precomputed audio features
            prev_motion_feat: (N, n_prev_motions, d_motion) previous transformation features
            prev_audio_feat: (N, n_prev_motions, feature_dim) previous audio features
            time_step: (N,) diffusion steps (if None, sampled uniformly)
            indicator: (N, L) optional indicator for valid motion data

        Returns:
            eps: noise applied to motion_feat
            motion_feat_target: predicted noise (or sample) of shape (N, L, d_motion)
            motion_feat: original motion features (detached)
            audio_feat_saved: extracted audio features (detached)
        """
        batch_size = motion_feat.shape[0]

        if audio_or_feat.ndim == 2:
            # Extract audio features from raw audio
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
            # print(audio_feat_saved.shape)
            # audio_feat_saved = self.audio_encoder(audio_or_feat).last_hidden_state[:, 1:, :] 
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        audio_feat = audio_feat_saved.clone()

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)
        if prev_audio_feat is None:
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)  # list of ints

        # Forward diffusion: add noise to the true motion feature.
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]  # (N,)
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        # Denoising network predicts the noise (or target transformation) conditioned solely on audio.
        motion_feat_target = self.denoising_net(motion_feat_noisy, audio_feat, prev_motion_feat, prev_audio_feat, time_step, indicator)

        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions
        # Strategy: extract features and then downsample by interpolation.
        # print(audio.shape, pad_audio(audio).shape)
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num * 2).last_hidden_state  # (N, 2L, 768)
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')  # (N, 768, L)
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)
        audio_feat = self.audio_feature_map(hidden_states)  # (N, L, feature_dim)
        return audio_feat
    
    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, motion_at_T=None,
               indicator=None, flexibility=0, dynamic_threshold=None, ret_traj=False):
        """
        Reverse diffusion (sampling) process conditioned on audio.
        """
        batch_size = audio_or_feat.shape[0]

        if audio_or_feat.ndim == 2:
            audio_feat = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)
        if prev_audio_feat is None:
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(self.device)

        traj = {self.diffusion_sched.num_steps: motion_at_T}

        for t in range(self.diffusion_sched.num_steps, 0, -1):
            z = torch.randn_like(motion_at_T) if t > 1 else torch.zeros_like(motion_at_T)
            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            step_in = torch.tensor([t] * batch_size, device=self.device)
            motion_pred = self.denoising_net(motion_at_t, audio_feat, prev_motion_feat, prev_audio_feat, step_in, indicator)

            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = motion_pred[:, -self.n_motions:].reshape(batch_size, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max).unsqueeze(-1).unsqueeze(-1)
                motion_pred = torch.clamp(motion_pred, min=-s, max=s)

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * motion_pred) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * motion_pred + sigma * z
            else:
                raise ValueError('Unknown target type: {}'.format(self.target))
            
            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat
        else:
            return traj[0], motion_at_T, audio_feat


class DenoisingNetwork(nn.Module):
    def __init__(self, args, device='cuda'):
        super().__init__()

        # The output (and input) motion feature dimension is 72, as defined above.
        self.motion_feat_dim = 9 + 3 + 204   # 216
        self.use_indicator = args.use_indicator

        # Transformer parameters
        self.architecture = args.architecture
        self.feature_dim = args.feature_dim
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.mlp_ratio = args.mlp_ratio
        self.align_mask_width = args.align_mask_width
        self.use_learnable_pe = not args.no_use_learnable_pe

        # Sequence lengths for previous and current motions.
        self.n_prev_motions = args.n_prev_motions
        self.n_motions = args.n_motions

        # Temporal embedding for the diffusion time step.
        self.TE = PositionalEncoding(self.feature_dim, max_len=args.n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        if self.use_learnable_pe:
            self.PE = nn.Parameter(torch.randn(1, self.n_prev_motions + self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        # Transformer decoder
        if self.architecture == 'decoder':
            input_dim = self.motion_feat_dim + (1 if self.use_indicator else 0)
            self.feature_proj = nn.Linear(input_dim, self.feature_dim)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.feature_dim,
                nhead=self.n_heads,
                dim_feedforward=self.mlp_ratio * self.feature_dim,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
            if self.align_mask_width > 0:
                motion_len = self.n_prev_motions + self.n_motions
                alignment_mask = enc_dec_mask(motion_len, motion_len, 1, self.align_mask_width - 1)
                # alignment_mask = F.pad(alignment_mask, (0, 0, 1, 0), value=False)
                self.register_buffer('alignment_mask', alignment_mask)
            else:
                self.alignment_mask = None
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Motion decoder: maps the transformer output to 72 transformation parameters.
        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim)
        )

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_motion) noisy motion features (current frame)
            audio_feat: (N, L', feature_dim) audio features (from audio encoder)
            prev_motion_feat: (N, L_prev, d_motion) previous motion features
            prev_audio_feat: (N, L_prev, feature_dim) previous audio features
            step: (N,) diffusion time steps
            indicator: (N, L) optional mask

        Returns:
            motion_feat_target: (N, L_current, d_motion) predicted noise or target transformation
        """
        # Compute diffusion time step embedding.
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)  # (N, 1, feature_dim)

        if indicator is not None:
            indicator = torch.cat([torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                                   indicator], dim=1)  # (N, L_prev + L_current)
            indicator = indicator.unsqueeze(-1)  # (N, L_prev + L_current, 1)

        # Concatenate previous and current noisy motion features.
        if self.architecture == 'decoder':
            feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)  # (N, total_length, d_motion)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)

        feats_in = self.feature_proj(feats_in)  # (N, total_length, feature_dim)

        # Add time step embedding.
        feats_in = feats_in + diff_step_embedding

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        # Use audio features as memory in the transformer.
        if self.architecture == 'decoder':
            audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
            #print(audio_feat_in.shape)
            feat_out = self.transformer(feats_in, audio_feat_in, memory_mask=self.alignment_mask)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Decode only the current part (after the previous motions) to predict the transformation parameters.
        motion_feat_target = self.motion_dec(feat_out[:, self.n_prev_motions:])
        return motion_feat_target

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or sample from DiffTalkingHead diffusion model")
    parser.add_argument("--target", choices=["noise", "sample"], default="sample")
    parser.add_argument("--architecture", type=str, default="decoder")
    parser.add_argument("--audio_model", type=str, default="hubert")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--n_prev_motions", type=int, default=10)
    parser.add_argument("--n_motions", type=int, default=50)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--n_diff_steps", type=int, default=500)
    parser.add_argument("--diff_schedule", type=str, default="cosine")
    parser.add_argument("--no_use_learnable_pe", type=bool, default=False)
    parser.add_argument("--use_indicator", type=bool, default=True)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--align_mask_width", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda", help='cuda or cpu')
    args = parser.parse_args()

    # pick device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # instantiate model
    model = DiffTalkingHead(args, device=device)
    model.eval()  # set to eval for sampling

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ---- Dummy inputs for a batch of size N=2 ----
    N = 2
    L_motion = args.n_motions                # sequence length of motions
    d_motion = model.motion_feat_dim   # 216

    L_audio = 32000
    feat_dim = args.feature_dim        # e.g. 512

    # ground‐truth motion
    motion_feat = torch.randn(N, L_motion, d_motion, device=device)

    # precomputed audio features (N, L_audio=L, feature_dim)
    # audio_feat = torch.randn(N, L, feat_dim, device=device)
    audio_feat = torch.randn(N, L_audio, device=device)

    # optional indicator mask for valid motion frames
    indicator = torch.ones(N, L_motion, dtype=torch.bool, device=device)

    # ---- Forward pass ----
    eps, motion_target, orig_motion, audio_saved = model.forward(
        motion_feat=motion_feat,
        audio_or_feat=audio_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        time_step=None,
        indicator=indicator
    )

    print("Forward outputs:")
    print("eps shape:", eps.shape)            # (N, L, d_motion)
    print("motion_target shape:", motion_target.shape)   # (N, L, d_motion)
    print("orig_motion shape:", orig_motion.shape)     # (N, L, d_motion)
    print("audio_saved shape:", audio_saved.shape)     # (N, L, feature_dim)


    # ---- Sampling (reverse diffusion) ----
    sampled_motion, motion_T, audio_feat_saved = model.sample(
        audio_or_feat=audio_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        motion_at_T=None,
        indicator=indicator,
        flexibility=0.0,
        dynamic_threshold=None,
        ret_traj=False
    )
    print("Sample outputs:")
    print("sampled_motion shape:", sampled_motion.shape)  # (N, L, d_motion)
    print("motion_T shape:", motion_T.shape)        # (N, L, d_motion)
    print("audio_feat_saved:", audio_feat_saved.shape) # (N, L, feature_dim)

