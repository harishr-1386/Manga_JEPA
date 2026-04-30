import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open_clip
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from collections import Counter

load_dotenv()

EMBEDDINGS_DIR  = Path(os.getenv('EMBEDDINGS_DIR'))
OUTPUT_DIR      = Path('data/action_labels')
MODELS_DIR      = Path(os.getenv('MODELS_DIR'))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PROJECTION_PATH = MODELS_DIR / 'vjepa_to_clip_projection.pt'

VOLUMES = ['old_boy_vol01', 'old_boy_vol02', 'old_boy_vol03']

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

action_to_idx = {a: i for i, a in enumerate(ACTION_SHORT)}


class VJEPAProjection(nn.Module):
    """
    Projects V-JEPA 2 1024-dim embeddings to CLIP 512-dim space.
    Two-layer MLP with layer normalization.
    """
    def __init__(self, input_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)
        return projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)


class PanelSequenceDataset(Dataset):
    """
    Dataset of (vjepa_sequence_embedding, clip_text_embedding, action_idx) triples.
    Uses CLIP panel action labels as pseudo-labels via majority vote per sequence.
    """
    def __init__(
        self,
        vjepa_seq_embs: np.ndarray,
        seq_metadata: list[dict],
        clip_panel_labels: dict,
        clip_text_embeds: np.ndarray,
    ):
        self.vjepa_embs       = torch.tensor(vjepa_seq_embs, dtype=torch.float32)
        self.clip_text_embeds = torch.tensor(clip_text_embeds, dtype=torch.float32)
        self.targets          = []

        for meta in seq_metadata:
            action_votes = {}
            for panel_id in meta['panel_ids']:
                if panel_id in clip_panel_labels:
                    action = clip_panel_labels[panel_id]['action']
                    action_votes[action] = action_votes.get(action, 0) + 1
            if action_votes:
                majority_action = max(action_votes, key=action_votes.get)
                action_idx      = action_to_idx.get(majority_action, 4)
            else:
                action_idx = 4
            self.targets.append(action_idx)

        self.targets = torch.tensor(self.targets, dtype=torch.long)

        print(f'Dataset: {len(self.vjepa_embs)} sequences')
        print('Label distribution:')
        dist = Counter(self.targets.numpy())
        for idx, count in sorted(dist.items()):
            print(f'  {ACTION_SHORT[idx]}: {count}')

    def __len__(self):
        return len(self.vjepa_embs)

    def __getitem__(self, idx):
        vjepa_emb       = self.vjepa_embs[idx]
        action_idx      = self.targets[idx]
        target_text_emb = self.clip_text_embeds[action_idx]
        return vjepa_emb, target_text_emb, action_idx


def train_projection(
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    temperature: float = 0.07,
    device: str = 'cuda',
):
    # Load CLIP text embeddings then clear GPU
    print('Loading CLIP text embeddings...')
    clip_m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_tok      = open_clip.get_tokenizer('ViT-B-32')
    clip_m.eval()
    with torch.no_grad():
        tokens     = clip_tok(ACTION_LABELS)
        text_feats = clip_m.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    clip_text_embeds = text_feats.cpu().numpy()
    del clip_m, clip_tok
    torch.cuda.empty_cache()
    print('CLIP text embeddings ready. GPU cleared.')

    # Load V-JEPA sequence embeddings + CLIP panel labels
    all_seq_embs    = []
    all_seq_meta    = []
    all_clip_labels = {}

    for vol in VOLUMES:
        seq_emb_path  = OUTPUT_DIR / f'{vol}_vjepa_seq_embeddings.npy'
        seq_meta_path = OUTPUT_DIR / f'{vol}_vjepa_seq_metadata.json'
        clip_path     = OUTPUT_DIR / f'{vol}_clip_actions.json'

        if not seq_emb_path.exists():
            print(f'No sequence embeddings for {vol}, skipping.')
            continue

        embs = np.load(seq_emb_path)
        with open(seq_meta_path) as f:
            meta = json.load(f)
        with open(clip_path) as f:
            clip_labels = json.load(f)

        all_seq_embs.append(embs)
        all_seq_meta.extend(meta)
        all_clip_labels.update(clip_labels)
        print(f'{vol}: {len(embs)} sequences loaded')

    if not all_seq_embs:
        print('No sequence embeddings found. Run sequential_encoder.py first.')
        return

    all_seq_embs = np.concatenate(all_seq_embs, axis=0)
    print(f'\nTotal sequences: {len(all_seq_embs)}')

    # Build dataset
    dataset    = PanelSequenceDataset(
        all_seq_embs, all_seq_meta, all_clip_labels, clip_text_embeds
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Soft class weights using square root dampening
    label_counts  = Counter(dataset.targets.numpy())
    total_samples = len(dataset)
    class_weights = torch.zeros(len(ACTION_SHORT))
    for idx in range(len(ACTION_SHORT)):
        count = label_counts.get(idx, 1)
        class_weights[idx] = (total_samples / (len(label_counts) * count)) ** 0.5
    class_weights = class_weights.to(device)
    print(f'\nClass weights (sqrt-dampened):')
    for i, w in enumerate(class_weights):
        print(f'  {ACTION_SHORT[i]}: {w.item():.2f}')

    # Build model
    model     = VJEPAProjection(input_dim=1024, output_dim=512).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    clip_text_tensor = torch.tensor(clip_text_embeds, dtype=torch.float32).to(device)
    ce_loss_fn       = nn.CrossEntropyLoss(weight=class_weights)

    print(f'\nTraining: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
    print(f'Device: {device} | Epochs: {epochs} | Batch: {batch_size} | '
          f'LR: {lr} | Temp: {temperature}')

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_agree = 0
        epoch_total = 0

        for vjepa_emb, target_text_emb, action_idx in dataloader:
            vjepa_emb       = vjepa_emb.to(device)
            target_text_emb = target_text_emb.to(device)
            action_idx      = action_idx.to(device)

            projected = model(vjepa_emb)

            # Temperature-scaled cross-entropy loss
            scores_all = projected @ clip_text_tensor.T / temperature
            ce_loss    = ce_loss_fn(scores_all, action_idx)

            # Cosine similarity loss against target text embedding
            cos_sims = (projected * target_text_emb).sum(dim=-1)
            cos_loss = (1 - cos_sims).mean()

            # Combined loss — equal weight
            loss = 0.5 * ce_loss + 0.5 * cos_loss

            # Accuracy
            with torch.no_grad():
                pred_idx    = scores_all.argmax(dim=-1)
                agree       = (pred_idx == action_idx).sum().item()
                epoch_agree += agree
                epoch_total += len(action_idx)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_agree / epoch_total if epoch_total > 0 else 0

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), PROJECTION_PATH)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:>3}/{epochs} | '
                  f'loss: {avg_loss:.4f} | '
                  f'accuracy: {accuracy*100:.1f}% | '
                  f'lr: {scheduler.get_last_lr()[0]:.6f}')

    print(f'\nBest loss: {best_loss:.4f}')
    print(f'Projection saved to {PROJECTION_PATH}')
    return model


def evaluate_projection(device: str = 'cuda'):
    """Evaluate trained projection against CLIP labels."""
    if not PROJECTION_PATH.exists():
        print('No trained projection found.')
        return

    # Load CLIP text embeddings
    clip_m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_tok      = open_clip.get_tokenizer('ViT-B-32')
    clip_m.eval()
    with torch.no_grad():
        tokens     = clip_tok(ACTION_LABELS)
        text_feats = clip_m.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    clip_text_embeds = text_feats.cpu().numpy()
    del clip_m, clip_tok
    torch.cuda.empty_cache()

    model = VJEPAProjection().to(device)
    model.load_state_dict(torch.load(PROJECTION_PATH, map_location=device))
    model.eval()

    clip_text_tensor = torch.tensor(clip_text_embeds, dtype=torch.float32).to(device)

    print('\nEvaluation: Projected V-JEPA 2 vs CLIP labels')
    print('=' * 55)

    total_agree  = 0
    total_panels = 0

    for vol in VOLUMES:
        seq_emb_path  = OUTPUT_DIR / f'{vol}_vjepa_seq_embeddings.npy'
        seq_meta_path = OUTPUT_DIR / f'{vol}_vjepa_seq_metadata.json'
        clip_path     = OUTPUT_DIR / f'{vol}_clip_actions.json'

        if not seq_emb_path.exists():
            continue

        embs = np.load(seq_emb_path)
        with open(seq_meta_path) as f:
            meta = json.load(f)
        with open(clip_path) as f:
            clip_labels = json.load(f)

        emb_tensor = torch.tensor(embs, dtype=torch.float32).to(device)
        with torch.no_grad():
            projected = model(emb_tensor)
            scores    = projected @ clip_text_tensor.T
            pred_idx  = scores.argmax(dim=-1).cpu().numpy()

        agree     = 0
        total     = 0
        proj_dist = Counter()

        for i, m in enumerate(meta):
            proj_action = ACTION_SHORT[pred_idx[i]]
            proj_dist[proj_action] += 1
            for panel_id in m['panel_ids']:
                if panel_id in clip_labels:
                    clip_action = clip_labels[panel_id]['action']
                    total      += 1
                    if clip_action == proj_action:
                        agree += 1

        agree_rate    = agree / total if total > 0 else 0
        total_agree  += agree
        total_panels += total

        print(f'\n{vol}:')
        print(f'  Agreement with CLIP: {agree_rate:.3f} ({agree}/{total})')
        print(f'  Projected distribution (top 5):')
        for action, count in proj_dist.most_common(5):
            print(f'    {action}: {count}')

    overall = total_agree / total_panels if total_panels > 0 else 0
    print(f'\nOverall agreement: {overall:.3f} ({total_agree}/{total_panels})')
    print(f'Baseline (naive mean-pool): 0.017')
    print(f'Unweighted projection:      0.339')
    print(f'Weighted projection:        0.110')
    print(f'This run:                   {overall:.3f}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    missing = [
        vol for vol in VOLUMES
        if not (OUTPUT_DIR / f'{vol}_vjepa_seq_embeddings.npy').exists()
    ]
    if missing:
        print(f'Missing sequence embeddings for: {missing}')
        print('Run sequential_encoder.py first.')
    else:
        print('Training projection layer...')
        train_projection(epochs=50, device=device)
        print('\nEvaluating...')
        evaluate_projection(device=device)
