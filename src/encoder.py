import torch
import sys
import os
import types
import importlib.util
from dotenv import load_dotenv

load_dotenv()

VJEPA2_HUB_PATH = os.getenv('VJEPA2_HUB_PATH')
WEIGHTS_PATH    = os.getenv('VJEPA2_WEIGHTS')

# Register vjepa2's src package in sys.modules BEFORE importing
# This prevents our own src package from shadowing vjepa2's internal imports
_vjepa_src = types.ModuleType('src')
_vjepa_src.__path__ = [os.path.join(VJEPA2_HUB_PATH, 'src')]
_vjepa_src.__package__ = 'src'
sys.modules['src'] = _vjepa_src

# Also add vjepa2 root to path so sub-imports resolve
if VJEPA2_HUB_PATH not in sys.path:
    sys.path.insert(0, VJEPA2_HUB_PATH)

from src.models.vision_transformer import vit_large


def load_encoder(device='cuda'):
    model = vit_large(
        img_size=256,
        patch_size=16,
        num_frames=8,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,
    )
    ckpt          = torch.load(WEIGHTS_PATH, map_location='cpu')
    encoder_state = ckpt['encoder']
    cleaned       = {k.replace('module.backbone.', ''): v for k, v in encoder_state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    model.to(device)
    print(f'Encoder loaded on {device} | {sum(p.numel() for p in model.parameters())/1e6:.0f}M params')
    return model


def encode_frames(model, frames: torch.Tensor, device='cuda') -> torch.Tensor:
    """
    frames: (B, C, T, H, W)
    returns: (B, N, D)
    """
    with torch.no_grad():
        frames     = frames.to(device)
        embeddings = model(frames)
    return embeddings


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_encoder(device)

    dummy = torch.randn(1, 3, 8, 256, 256)
    out   = encode_frames(model, dummy, device)
    print(f'Input:  {dummy.shape}')
    print(f'Output: {out.shape}')
