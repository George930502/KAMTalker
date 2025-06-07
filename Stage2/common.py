import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Vanilla sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        args:
            x shape: (batch, seq_len, d_model).
        '''
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
def enc_dec_mask(T, S, frame_width=2, expansion=0, device='cuda'):
    '''
    Ensures that each decoder token can only attend to specific encoder tokens.
    args:
        T: Number of target sequence tokens (decoder length).
        S: Number of source sequence tokens (encoder length). 
        frame_width (default = 2): Determines how many tokens are attended per step.
        expansion (default = 0): Expands the mask region, allowing more context.
    '''
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width): (i + expansion + 1) * frame_width] = 0
    return (mask == 1).to(device=device)

def pad_audio(audio, audio_unit=320, pad_threshold=80):
    """
    Args:
        audio: Tensor of shape (N, T)
        audio_unit: hop size, usually 320 for 50fps
        pad_threshold: threshold for additional padding

    Returns:
        Padded audio: (N, T')
    """
    _, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    
    if side_len > 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = F.pad(audio, (1, 1), mode='replicate')
            
    return audio