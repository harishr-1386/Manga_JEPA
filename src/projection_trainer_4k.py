import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open_clip
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from collections import Counter

load_dotenv()

EMBEDDINGS_DIR  = Path(os.getenv('EMBEDDINGS_DIR'))
OUTPUT_DIR      = Path('data/action_labels')
MODELS_DIR      = Path(os.getenv('MODELS_DIR'))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PROJECTION_PATH_4K = MODELS_DIR / 'vjepa_to_clip_projection_4k.pt'

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


class CombinedDataset(Dataset):
    """
    Combines sequence embeddings (997) + individual panel embeddings (4002)
    for a total of ~5000 training samples.
    All labeled via CLIP pseudo-labels.
    """
    def __init__(
        self,
        all_embs: np.ndarray,
        all_labels: list[int],
        clip_text_embeds: np.ndarray,
    ):
        self.embs             = torch.tensor(all_embs,        dtype=torch.float32)
        self.targets          = torch.tensor(all_labels,      dtype=torch.long)
        self.clip_text_embeds = torch.tensor(clip_text_embeds, dtype=torch.float32)

        print(f'Dataset: {len(self.embs)} samples')
        print('Label distribution:')
        dist = Counter(all_labels)
        for idx in range(len(ACTION_SHORT)):
            count = dist.get(idx, 0)
            if count > 0:
                print(f'  {ACTION_SHORT[idx]}: {count}')

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        emb            = self.embs[idx]
        action_idx     = self.targets[idx]
        target_text_emb = self.clip_text_embeds[action_idx]
        return emb, target_text_emb, action_idx


def load_all_data():
    """
    Load sequence embeddings + individual panel embeddings across all volumes.
    Returns combined embeddings array and label list.
    """
    all_embs   = []
    all_labels = []
    clip_all   = {}

    for vol in VOLUMES:
        seq_emb_path   = OUTPUT_DIR / f'{vol}_vjepa_seq_embeddings.npy'
        seq_meta_path  = OUTPUT_DIR / f'{vol}_vjepa_seq_metadata.json'
        clip_path      = OUTPUT_DIR / f'{vol}_clip_actions.json'
        panel_emb_path = EMBEDDINGS_DIR / vol / 'embeddings.npy'
        panel_meta_path = EMBEDDINGS_DIR / vol / 'metadata.json'

        if not clip_path.exists():
            print(f'No CLIP labels for {vol}, skipping.')
            continue

        with open(clip_path) as f:
            clip_labels = json.load(f)
        clip_all.update(clip_labels)

        # --- Sequence embeddings ---
        if seq_emb_path.exists():
            seq_embs = np.load(seq_emb_path)
            with open(seq_meta_path) as f:
                seq_meta = json.load(f)

            for emb, meta in zip(seq_embs, seq_meta):
                # Majority vote from panels in sequence
                votes = {}
                for pid in meta['panel_ids']:
                    if pid in clip_labels:
                        a = clip_labels[pid]['action']
                        votes[a] = votes.get(a, 0) + 1
                if votes:
                    action     = max(votes, key=votes.get)
                    action_idx = action_to_idx.get(action, 4)
                    all_embs.append(emb)
                    all_labels.append(action_idx)

            print(f'{vol}: {len(seq_embs)} sequences loaded')

        # --- Individual panel embeddings ---
        if panel_emb_path.exists():
            panel_embs = np.load(panel_emb_path)
            with open(panel_meta_path) as f:
                panel_meta = json.load(f)

            added = 0
            for emb, meta in zip(panel_embs, panel_meta):
                pid = meta['panel_id']
                if pid in clip_labels:
                    action     = clip_labels[pid]['action']
                    action_idx = action_to_idx.get(action, 4)
                    all_embs.append(emb)
                    all_labels.append(action_idx)
                    added += 1

            print(f'{vol}: {added} panel embeddings loaded')

    print(f'\nTotal combined samples: {len(all_embs)}')
    return np.array(all_embs), all_labels


def train_4k(
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
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

    # Load all data
    all_embs, all_labels = load_all_data()

    # Soft class weights
    label_counts  = Counter(all_labels)
    total_samples = len(all_labels)
    class_weights = torch.zeros(len(ACTION_SHORT))
    for idx in range(len(ACTION_SHORT)):
        count = label_counts.get(idx, 1)
        class_weights[idx] = (total_samples / (len(label_counts) * count)) ** 0.5
    class_weights = class_weights.to(device)

    print('\nClass weights (sqrt-dampened):')
    for i, w in enumerate(class_weights):
        if label_counts.get(i, 0) > 0:
            print(f'  {ACTION_SHORT[i]}: {w.item():.2f} (n={label_counts.get(i,0)})')

    # Build dataset + dataloader
    dataset    = CombinedDataset(all_embs, all_labels, clip_text_embeds)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Build model
    model      = VJEPAProjection(input_dim=1024, output_dim=512).to(device)
    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    clip_text_tensor = torch.tensor(clip_text_embeds, dtype=torch.float32).to(device)

    print(f'\nTraining 4K projection: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
    print(f'Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}')

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_agree = 0
        epoch_total = 0

        for emb, target_text_emb, action_idx in dataloader:
            emb             = emb.to(device)
            target_text_emb = target_text_emb.to(device)
            action_idx      = action_idx.to(device)

            projected  = model(emb)
            scores_all = projected @ clip_text_tensor.T / temperature
            ce_loss    = ce_loss_fn(scores_all, action_idx)
            cos_loss   = (1 - (projected * target_text_emb).sum(dim=-1)).mean()
            loss       = 0.5 * ce_loss + 0.5 * cos_loss

            with torch.no_grad():
                pred_idx    = scores_all.argmax(dim=-1)
                epoch_agree += (pred_idx == action_idx).sum().item()
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
            torch.save(model.state_dict(), PROJECTION_PATH_4K)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:>3}/{epochs} | '
                  f'loss: {avg_loss:.4f} | '
                  f'accuracy: {accuracy*100:.1f}% | '
                  f'lr: {scheduler.get_last_lr()[0]:.6f}')

    print(f'\nBest loss: {best_loss:.4f}')
    print(f'Projection saved to {PROJECTION_PATH_4K}')
    return model


def evaluate_4k(device: str = 'cuda'):
    if not PROJECTION_PATH_4K.exists():
        print('No 4K projection found.')
        return

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
    model.load_state_dict(torch.load(PROJECTION_PATH_4K, map_location=device))
    model.eval()

    clip_text_tensor = torch.tensor(clip_text_embeds, dtype=torch.float32).to(device)

    print('\nEvaluation: 4K Projected V-JEPA 2 vs CLIP labels')
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
    print(f'\nResults comparison:')
    print(f'  Baseline (naive):         0.017')
    print(f'  997 seq unweighted:       0.339')
    print(f'  997 seq fully weighted:   0.110')
    print(f'  997 seq soft weighted:    0.316')
    print(f'  4K combined soft weighted:{overall:.3f}  <-- this run')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print('Training 4K projection layer (sequences + panels)...')
    train_4k(epochs=50, device=device)
    print('\nEvaluating...')
    evaluate_4k(device=device)
