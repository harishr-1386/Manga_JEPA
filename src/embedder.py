import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import sys
import os
from dotenv import load_dotenv
import types


load_dotenv()

VJEPA2_HUB_PATH = os.getenv('VJEPA2_HUB_PATH')
WEIGHTS_PATH = os.getenv('VJEPA2_WEIGHTS')
PANELS_DIR = Path(os.getenv('PANELS_DIR'))
EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))

# Register vjepa2 src 
_vjepa_src          = types.ModuleType('src')
_vjepa_src.__path__ = [os.path.join(VJEPA2_HUB_PATH, 'src')]
_vjepa_src.__package__ = 'src'
sys.modules['src']  = _vjepa_src

if VJEPA2_HUB_PATH not in sys.path:
    sys.path.insert(0, VJEPA2_HUB_PATH)

from src.models.vision_transformer import vit_large

#sys.path.insert(0, VJEPA2_HUB_PATH)
#from src.models.vision_transformer import vit_large


EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# V-JEPA 2 expects 8 frames at 256x256
NUM_FRAMES = 8
IMG_SIZE = 256

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_encoder(device='cuda'):
    model = vit_large(
        img_size=IMG_SIZE,
        patch_size=16,
        num_frames=NUM_FRAMES,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,
    )
    ckpt = torch.load(WEIGHTS_PATH, map_location='cpu')
    cleaned = {k.replace('module.backbone.', ''): v
               for k, v in ckpt['encoder'].items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval().to(device)
    return model


def panel_to_clip(panel_path: str) -> torch.Tensor:
    """
    Turn a single panel image into an 8-frame clip.
    We repeat the panel 8 times — V-JEPA 2 sees it as a static 'video'.
    Returns: (1, 3, 8, 256, 256)
    """
    img = Image.open(panel_path).convert('RGB')
    frame = transform(img)                    # (3, 256, 256)
    clip = frame.unsqueeze(1).repeat(1, NUM_FRAMES, 1, 1)  # (3, 8, 256, 256)
    return clip.unsqueeze(0)                  # (1, 3, 8, 256, 256)


def embed_panels(manga_name: str, device='cuda', batch_size=8):
    """
    Encode all panels for a manga, save embeddings + metadata to disk.
    """
    panel_dir = PANELS_DIR / manga_name
    panel_files = sorted(panel_dir.glob('*.jpg'))

    if not panel_files:
        raise ValueError(f'No panels found in {panel_dir}')

    print(f'Encoding {len(panel_files)} panels for {manga_name}...')
    model = load_encoder(device)

    all_embeddings = []
    all_metadata = []

    for i in tqdm(range(0, len(panel_files), batch_size)):
        batch_files = panel_files[i:i + batch_size]
        clips = torch.cat([panel_to_clip(str(f)) for f in batch_files], dim=0)
        clips = clips.to(device)

        with torch.no_grad():
            # Output: (B, N_tokens, D) — mean pool tokens -> (B, D)
            embeddings = model(clips)
            embeddings = embeddings.mean(dim=1)  # (B, 1024)

        all_embeddings.append(embeddings.cpu().float().numpy())

        for f in batch_files:
            name = f.stem  # e.g. page0005_panel0002
            parts = name.split('_')
            page = int(parts[0].replace('page', ''))
            panel = int(parts[1].replace('panel', ''))
            all_metadata.append({
                'manga': manga_name,
                'page': page,
                'panel': panel,
                'path': str(f),
                'panel_id': name,
            })

    embeddings_np = np.concatenate(all_embeddings, axis=0)

    # Save
    out_path = EMBEDDINGS_DIR / manga_name
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / 'embeddings.npy', embeddings_np)
    with open(out_path / 'metadata.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f'\nDone. Embeddings shape: {embeddings_np.shape}')
    print(f'Saved to {out_path}')
    return embeddings_np, all_metadata


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings, metadata = embed_panels('old_boy_vol01', device=device)
    print(f'\nSample metadata[0]: {metadata[0]}')
    print(f'Sample embedding shape: {embeddings[0].shape}')