import os
import json
import torch
import numpy as np
import open_clip
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

PANELS_DIR     = Path(os.getenv('PANELS_DIR'))
EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
OUTPUT_DIR     = Path('data/action_labels')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOLUMES = ['old_boy_vol01', 'old_boy_vol02', 'old_boy_vol03']

# Action label set — designed for manga noir like Old Boy
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

# Short display labels
ACTION_SHORT = [
    'fighting',
    'running',
    'talking',
    'sad',
    'angry',
    'eating',
    'dark scene',
    'walking alone',
    'flashback',
    'revelation',
    'captured',
    'crowd',
]

# Load CLIP once
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.eval()

# Pre-encode all action labels as text embeddings
with torch.no_grad():
    tokens      = clip_tokenizer(ACTION_LABELS)
    text_feats  = clip_model.encode_text(tokens)
    text_feats  = text_feats / text_feats.norm(dim=-1, keepdim=True)
    TEXT_EMBEDS = text_feats.cpu().numpy()  # (n_actions, 512)


def classify_panel_clip(image_path: str) -> dict:
    """Classify a single panel using CLIP zero-shot."""
    img  = clip_preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        img_feat = clip_model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        img_np   = img_feat.cpu().numpy()[0]

    scores     = img_np @ TEXT_EMBEDS.T  # (n_actions,)
    top_idx    = int(np.argmax(scores))
    top_score  = float(scores[top_idx])

    # Top 3 actions
    top3_idx   = np.argsort(scores)[::-1][:3]

    return {
        'action':       ACTION_SHORT[top_idx],
        'action_full':  ACTION_LABELS[top_idx],
        'confidence':   round(top_score, 4),
        'top3': [
            {'action': ACTION_SHORT[i], 'score': round(float(scores[i]), 4)}
            for i in top3_idx
        ],
    }


def run_clip_action_detection(manga_name: str) -> dict:
    """Run CLIP action detection on all panels of a manga volume."""
    with open(EMBEDDINGS_DIR / manga_name / 'metadata.json') as f:
        metadata = json.load(f)

    results = {}
    for meta in tqdm(metadata, desc=f'CLIP action detection: {manga_name}'):
        panel_id   = meta['panel_id']
        image_path = str(PANELS_DIR / manga_name / f'{panel_id}.jpg')
        if not Path(image_path).exists():
            continue
        result             = classify_panel_clip(image_path)
        results[panel_id]  = {**meta, **result}

    # Save
    out_path = OUTPUT_DIR / f'{manga_name}_clip_actions.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print distribution
    from collections import Counter
    dist = Counter(r['action'] for r in results.values())
    print(f'\nAction distribution for {manga_name}:')
    for action, count in dist.most_common():
        pct = count / len(results) * 100
        print(f'  {action:<20} {count:>5} ({pct:.1f}%)')

    return results


def get_panels_by_action(
    manga_name: str,
    action: str,
    top_k: int = 10,
) -> list[dict]:
    """Retrieve panels by action label, sorted by confidence."""
    actions_path = OUTPUT_DIR / f'{manga_name}_clip_actions.json'
    if not actions_path.exists():
        raise FileNotFoundError(f'Run action detection first for {manga_name}')

    with open(actions_path) as f:
        all_actions = json.load(f)

    matched = [
        v for v in all_actions.values()
        if v['action'] == action or action in v['action']
    ]
    matched.sort(key=lambda x: x['confidence'], reverse=True)
    return matched[:top_k]


if __name__ == '__main__':
    for vol in VOLUMES:
        print(f'\n{"="*50}')
        results = run_clip_action_detection(vol)
        print(f'Total panels processed: {len(results)}')