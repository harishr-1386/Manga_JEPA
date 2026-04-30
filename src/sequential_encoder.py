import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from dotenv import load_dotenv
from src.encoder import load_encoder

load_dotenv()

PANELS_DIR     = Path(os.getenv('PANELS_DIR'))
EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
OUTPUT_DIR     = Path('data/action_labels')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOLUMES       = ['old_boy_vol01', 'old_boy_vol02', 'old_boy_vol03']
SEQUENCE_LEN  = 8   # number of consecutive panels per clip
STRIDE        = 4   # overlap between sequences

# Same action labels as CLIP for comparison
ACTION_LABELS = [
    'two people fighting or brawling',
    'a person running or chasing',
    'people talking or having a conversation',
    'a person looking sad or crying',
    'a person looking angry or threatening',
    'a person eating or drinking',
    'a dark or ominous scene',
    'a person walking alone',
    'a flashback or memory scene',
    'a shocking or dramatic revelation',
    'a person being restrained or captured',
    'a crowd or group of people',
]

ACTION_SHORT = [
    'fighting', 'running', 'talking', 'sad', 'angry',
    'eating', 'dark scene', 'walking alone', 'flashback',
    'revelation', 'captured', 'crowd',
]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def panels_to_clip(panel_paths: list[str], device: str = 'cuda') -> torch.Tensor:
    """
    Load N panel images and stack as a video clip.
    Pads with last frame if fewer than SEQUENCE_LEN panels available.
    Returns: (1, 3, SEQUENCE_LEN, 256, 256)
    """
    frames = []
    for path in panel_paths:
        try:
            img = Image.open(path).convert('RGB')
            frames.append(transform(img))
        except Exception:
            if frames:
                frames.append(frames[-1])  # repeat last frame
            else:
                frames.append(torch.zeros(3, 256, 256))

    # Pad to SEQUENCE_LEN
    while len(frames) < SEQUENCE_LEN:
        frames.append(frames[-1])
    frames = frames[:SEQUENCE_LEN]

    clip = torch.stack(frames, dim=1)  # (3, T, 256, 256)
    return clip.unsqueeze(0).to(device)  # (1, 3, T, 256, 256)


def build_sequence_embeddings(
    manga_name: str,
    model: torch.nn.Module,
    device: str = 'cuda',
) -> tuple[np.ndarray, list[dict]]:
    """
    Encode all consecutive panel sequences for a manga volume.
    Returns embeddings and sequence metadata.
    """
    with open(EMBEDDINGS_DIR / manga_name / 'metadata.json') as f:
        metadata = json.load(f)

    panel_ids   = [m['panel_id'] for m in metadata]
    panel_paths = [
        str(PANELS_DIR / manga_name / f'{pid}.jpg')
        for pid in panel_ids
    ]

    seq_embeddings = []
    seq_metadata   = []

    indices = range(0, len(panel_ids) - SEQUENCE_LEN + 1, STRIDE)

    for start_idx in tqdm(indices, desc=f'V-JEPA 2 sequences: {manga_name}'):
        end_idx    = start_idx + SEQUENCE_LEN
        seq_paths  = panel_paths[start_idx:end_idx]
        seq_pids   = panel_ids[start_idx:end_idx]

        clip = panels_to_clip(seq_paths, device)
        with torch.no_grad():
            emb = model(clip).mean(dim=1).cpu().numpy()[0]  # (1024,)

        seq_embeddings.append(emb)
        seq_metadata.append({
            'manga':        manga_name,
            'start_panel':  seq_pids[0],
            'end_panel':    seq_pids[-1],
            'start_idx':    start_idx,
            'end_idx':      end_idx - 1,
            'panel_ids':    seq_pids,
        })

    return np.array(seq_embeddings), seq_metadata


def classify_sequence_vjepa(
    seq_embedding: np.ndarray,
    clip_text_embeds: np.ndarray,
) -> dict:
    """
    Classify a sequence embedding against action labels.
    Uses precomputed CLIP text embeddings for label matching.
    Note: this is cross-modal — V-JEPA visual vs CLIP text.
    The comparison is approximate since spaces differ.
    """
    seq_norm   = seq_embedding / (np.linalg.norm(seq_embedding) + 1e-8)
    # Project V-JEPA 1024-dim to 512-dim via mean pooling pairs
    projected  = seq_norm.reshape(512, 2).mean(axis=1)
    projected  = projected / (np.linalg.norm(projected) + 1e-8)

    scores     = projected @ clip_text_embeds.T
    top_idx    = int(np.argmax(scores))
    top_score  = float(scores[top_idx])
    top3_idx   = np.argsort(scores)[::-1][:3]

    return {
        'action':      ACTION_SHORT[top_idx],
        'action_full': ACTION_LABELS[top_idx],
        'confidence':  round(top_score, 4),
        'top3': [
            {'action': ACTION_SHORT[i], 'score': round(float(scores[i]), 4)}
            for i in top3_idx
        ],
    }


def run_vjepa_action_detection(
    manga_name: str,
    model: torch.nn.Module,
    clip_text_embeds: np.ndarray,
    device: str = 'cuda',
) -> dict:
    """Run V-JEPA 2 sequential action detection on a manga volume."""
    seq_embeddings, seq_metadata = build_sequence_embeddings(
        manga_name, model, device
    )

    # Classify each sequence
    results = {}
    for emb, meta in zip(seq_embeddings, seq_metadata):
        classification = classify_sequence_vjepa(emb, clip_text_embeds)
        key            = f'{meta["start_panel"]}_to_{meta["end_panel"]}'
        results[key]   = {**meta, **classification}

    # Save
    out_path = OUTPUT_DIR / f'{manga_name}_vjepa_actions.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Distribution
    from collections import Counter
    dist = Counter(r['action'] for r in results.values())
    print(f'\nV-JEPA 2 action distribution for {manga_name}:')
    for action, count in dist.most_common():
        pct = count / len(results) * 100
        print(f'  {action:<20} {count:>5} ({pct:.1f}%)')

    return results


def compare_clip_vjepa(manga_name: str):
    """
    Compare CLIP per-panel vs V-JEPA 2 sequential action labels.
    Shows agreement rate and where they disagree.
    """
    clip_path  = OUTPUT_DIR / f'{manga_name}_clip_actions.json'
    vjepa_path = OUTPUT_DIR / f'{manga_name}_vjepa_actions.json'

    if not clip_path.exists() or not vjepa_path.exists():
        print('Run both detectors first.')
        return

    with open(clip_path)  as f: clip_labels  = json.load(f)
    with open(vjepa_path) as f: vjepa_labels = json.load(f)

    agree    = 0
    disagree = 0
    disagree_cases = []

    for seq_key, vjepa_result in vjepa_labels.items():
        # Check each panel in the sequence against its CLIP label
        for panel_id in vjepa_result['panel_ids']:
            if panel_id not in clip_labels:
                continue
            clip_action  = clip_labels[panel_id]['action']
            vjepa_action = vjepa_result['action']

            if clip_action == vjepa_action:
                agree += 1
            else:
                disagree += 1
                disagree_cases.append({
                    'panel_id':    panel_id,
                    'clip_action': clip_action,
                    'vjepa_action': vjepa_action,
                    'sequence':    seq_key,
                })

    total      = agree + disagree
    agree_rate = agree / total if total > 0 else 0

    print(f'\nCLIP vs V-JEPA 2 Action Agreement — {manga_name}')
    print(f'  Agreement rate: {agree_rate:.3f} ({agree}/{total})')
    print(f'  Disagreements:  {disagree}')

    if disagree_cases:
        print(f'\n  Top disagreement patterns:')
        from collections import Counter
        patterns = Counter(
            f'{d["clip_action"]} -> {d["vjepa_action"]}'
            for d in disagree_cases
        )
        for pattern, count in patterns.most_common(8):
            print(f'    {pattern}: {count}')

    return {
        'agree_rate':      round(agree_rate, 4),
        'agree':           agree,
        'disagree':        disagree,
        'disagree_cases':  disagree_cases[:20],
    }

torch.cuda.empty_cache()

if __name__ == '__main__':
    import open_clip as oc
    import torch

    # Load CLIP text embeddings for label matching
    print('Loading CLIP text embeddings...')
    clip_m, _, _ = oc.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_tok     = oc.get_tokenizer('ViT-B-32')
    clip_m.eval()
    with torch.no_grad():
        tokens     = clip_tok(ACTION_LABELS)
        text_feats = clip_m.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    clip_text_embeds = text_feats.cpu().numpy()


     # Explicitly unload CLIP from memory before loading V-JEPA 2
    del clip_m
    del clip_tok
    torch.cuda.empty_cache()
    print('CLIP unloaded from GPU.')

    # Load V-JEPA 2
    print('Loading V-JEPA 2...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_encoder(device)

    # Run on all volumes
    for vol in VOLUMES:
        print(f'\n{"="*55}')
        run_vjepa_action_detection(vol, model, clip_text_embeds, device)

    # Compare
    print(f'\n{"="*55}')
    print('CLIP vs V-JEPA 2 COMPARISON')
    print('='*55)
    for vol in VOLUMES:
        compare_clip_vjepa(vol)