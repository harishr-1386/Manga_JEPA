import os
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from dotenv import load_dotenv
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

load_dotenv()

EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
PANELS_DIR     = Path(os.getenv('PANELS_DIR'))
MANGA_NAME     = 'old_boy_vol01'
REGISTRY_PATH  = Path(f'data/labels/{MANGA_NAME}_character_registry.json')
OUTPUT_DIR     = Path('data/oldboy_cluster_eval')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHARACTERS  = ['Shinichi Goto', 'Takaaki Kakinuma', 'Eri']
CHAR_COLORS = {
    'Shinichi Goto':    (50,  100, 200),
    'Takaaki Kakinuma': (200, 60,  60),
    'Eri':              (60,  160, 80),
}
CLUSTER_COLORS = [
    (220, 50,  50),
    (50,  120, 220),
    (50,  180, 50),
    (220, 160, 50),
    (160, 50,  220),
    (50,  200, 200),
    (220, 50,  160),
    (100, 180, 80),
]
NOISE_COLOR = (80, 80, 80)


def get_fonts():
    try:
        return {
            'regular': ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12),
            'bold':    ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13),
            'title':   ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15),
            'small':   ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11),
        }
    except Exception:
        default = ImageFont.load_default()
        return {k: default for k in ['regular', 'bold', 'title', 'small']}


def cluster_purity(labels_true, labels_pred):
    total  = len(labels_true)
    purity = 0.0
    for cid in set(labels_pred):
        if cid == -1:
            continue
        mask  = labels_pred == cid
        most  = Counter(labels_true[mask]).most_common(1)[0][1]
        purity += most
    return round(purity / total, 4)


def per_character_recall(labels_true, labels_pred, char_to_int):
    recalls = {}
    for char, idx in char_to_int.items():
        char_mask   = labels_true == idx
        total_char  = char_mask.sum()
        if total_char == 0:
            recalls[char] = 0.0
            continue
        # Find the most common predicted cluster for this character
        char_preds  = labels_pred[char_mask]
        valid_preds = char_preds[char_preds != -1]
        if len(valid_preds) == 0:
            recalls[char] = 0.0
            continue
        best_cluster  = Counter(valid_preds).most_common(1)[0][0]
        in_best       = (valid_preds == best_cluster).sum()
        recalls[char] = round(float(in_best) / float(total_char), 4)
    return recalls


def load_labeled_embeddings():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    # Load embeddings and metadata
    clip_emb  = np.load(EMBEDDINGS_DIR / MANGA_NAME / 'clip_embeddings.npy')
    vjepa_emb = np.load(EMBEDDINGS_DIR / MANGA_NAME / 'embeddings.npy')
    with open(EMBEDDINGS_DIR / MANGA_NAME / 'metadata.json') as f:
        metadata = json.load(f)

    panel_ids       = [m['panel_id'] for m in metadata]
    panel_id_to_idx = {pid: i for i, pid in enumerate(panel_ids)}

    # Filter to manually labeled panels only
    valid_chars = [c for c in CHARACTERS if any(v == c for v in registry.values())]
    char_to_int = {c: i for i, c in enumerate(valid_chars)}

    labeled_indices = []
    labels_true     = []
    panel_id_list   = []

    for panel_id, char in registry.items():
        if char not in char_to_int:
            continue
        if panel_id not in panel_id_to_idx:
            continue
        labeled_indices.append(panel_id_to_idx[panel_id])
        labels_true.append(char_to_int[char])
        panel_id_list.append(panel_id)

    labels_true = np.array(labels_true)
    clip_labeled  = clip_emb[labeled_indices]
    vjepa_labeled = vjepa_emb[labeled_indices]

    return clip_labeled, vjepa_labeled, labels_true, char_to_int, panel_id_list, metadata


def run_clustering(embeddings, labels_true, char_to_int, encoder_name):
    normed      = normalize(embeddings)
    clusterer   = HDBSCAN(min_cluster_size=2, min_samples=1,
                          metric='euclidean', copy=True)
    labels_pred = clusterer.fit_predict(normed)

    n_clusters  = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    n_noise     = int((labels_pred == -1).sum())
    mask        = labels_pred != -1

    if mask.sum() < 3:
        return labels_pred, {
            'encoder': encoder_name, 'n_clusters': n_clusters,
            'n_noise': n_noise, 'purity': 0, 'ari': 0, 'nmi': 0,
            'per_char_recall': {}
        }

    purity  = cluster_purity(labels_true[mask], labels_pred[mask])
    ari     = adjusted_rand_score(labels_true[mask], labels_pred[mask])
    nmi     = normalized_mutual_info_score(labels_true[mask], labels_pred[mask])
    recalls = per_character_recall(labels_true, labels_pred, char_to_int)

    return labels_pred, {
        'encoder':          encoder_name,
        'n_clusters':       n_clusters,
        'n_noise':          n_noise,
        'purity':           purity,
        'ari':              round(float(ari),  4),
        'nmi':              round(float(nmi),  4),
        'per_char_recall':  recalls,
    }


def load_panel_thumb(panel_id, size=80):
    path = PANELS_DIR / MANGA_NAME / f'{panel_id}.jpg'
    try:
        img   = Image.open(path).convert('RGB')
        ratio = size / max(img.width, img.height)
        img   = img.resize((int(img.width * ratio), int(img.height * ratio)))
        canvas = Image.new('RGB', (size, size), (220, 220, 220))
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        return canvas
    except Exception:
        return Image.new('RGB', (size, size), (200, 200, 200))


def build_grid(panel_ids, labels_true, clip_preds, vjepa_preds,
               char_to_int, clip_metrics, vjepa_metrics):
    fonts     = get_fonts()
    THUMB     = 80
    BORDER    = 3
    CELL      = THUMB + BORDER * 2
    PADDING   = 12
    HEADER_H  = 60
    ROW_LABEL = 120
    N         = len(panel_ids)

    int_to_char = {v: k for k, v in char_to_int.items()}

    # Layout: one row per encoder + ground truth
    ROWS = ['Ground Truth', 'CLIP Clustering', 'V-JEPA 2 Clustering']
    total_w = ROW_LABEL + N * CELL + PADDING * 2
    total_h = HEADER_H + len(ROWS) * (CELL + 30) + PADDING * 2 + 120

    canvas = Image.new('RGB', (total_w, total_h), (248, 248, 248))
    draw   = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([0, 0, total_w, HEADER_H], fill=(20, 20, 20))
    draw.text((PADDING, 10), 'Old Boy Vol.1 — Character Clustering on Manual Labels',
              fill=(255, 255, 255), font=fonts['title'])
    draw.text((PADDING, 34),
              f'Labeled panels: {N}  |  Characters: {len(char_to_int)}  |  '
              f'Goto:{sum(labels_true==char_to_int.get("Shinichi Goto",99))}  '
              f'Kakinuma:{sum(labels_true==char_to_int.get("Takaaki Kakinuma",99))}  '
              f'Eri:{sum(labels_true==char_to_int.get("Eri",99))}',
              fill=(160, 160, 160), font=fonts['small'])

    row_preds = [labels_true, clip_preds, vjepa_preds]
    row_colors_fn = [
        lambda l: CHAR_COLORS.get(int_to_char.get(l, ''), (150, 150, 150)),
        lambda l: NOISE_COLOR if l == -1 else CLUSTER_COLORS[l % len(CLUSTER_COLORS)],
        lambda l: NOISE_COLOR if l == -1 else CLUSTER_COLORS[l % len(CLUSTER_COLORS)],
    ]

    row_subtitles = [
        'Ground Truth (manual labels)',
        f'CLIP  |  clusters:{clip_metrics["n_clusters"]}  noise:{clip_metrics["n_noise"]}  '
        f'purity:{clip_metrics["purity"]:.3f}  ARI:{clip_metrics["ari"]:.3f}',
        f'V-JEPA 2  |  clusters:{vjepa_metrics["n_clusters"]}  noise:{vjepa_metrics["n_noise"]}  '
        f'purity:{vjepa_metrics["purity"]:.3f}  ARI:{vjepa_metrics["ari"]:.3f}',
    ]

    for row_i, (row_name, preds, color_fn, subtitle) in enumerate(
        zip(ROWS, row_preds, row_colors_fn, row_subtitles)
    ):
        y_base = HEADER_H + PADDING + row_i * (CELL + 30)

        # Row label
        row_bg = (235, 240, 250) if row_i == 1 else (250, 235, 235) if row_i == 2 else (235, 250, 235)
        draw.rectangle([0, y_base, ROW_LABEL - 4, y_base + CELL + 28], fill=row_bg)
        draw.text((4, y_base + 4),  row_name, fill=(20, 20, 20), font=fonts['bold'])
        draw.text((4, y_base + 20), subtitle[:18], fill=(60, 60, 60), font=fonts['small'])

        # Panels
        x = ROW_LABEL + PADDING
        for j, (pid, pred) in enumerate(zip(panel_ids, preds)):
            color = color_fn(int(pred))

            # Border
            draw.rectangle([x, y_base, x + CELL - 1, y_base + CELL - 1], fill=color)

            # Thumbnail
            thumb = load_panel_thumb(pid, THUMB)
            canvas.paste(thumb, (x + BORDER, y_base + BORDER))

            # True label indicator (small dot in corner for non-GT rows)
            if row_i > 0:
                true_color = CHAR_COLORS.get(int_to_char.get(int(labels_true[j]), ''), (150, 150, 150))
                draw.rectangle([x + CELL - 10, y_base, x + CELL - 1, y_base + 10], fill=true_color)

            x += CELL

        # Subtitle below row
        draw.text((ROW_LABEL + PADDING, y_base + CELL + 4),
                  subtitle, fill=(60, 60, 60), font=fonts['small'])

    # Metrics summary
    y_metrics = HEADER_H + PADDING + len(ROWS) * (CELL + 30) + PADDING

    draw.rectangle([0, y_metrics, total_w, total_h], fill=(235, 235, 240))
    draw.text((PADDING, y_metrics + 8),
              'Per-Character Recall (fraction of character panels landing in best cluster):',
              fill=(20, 20, 20), font=fonts['bold'])

    y_t = y_metrics + 28
    draw.text((PADDING, y_t),
              f'{"Character":<22} {"CLIP Recall":>12} {"V-JEPA Recall":>14}',
              fill=(20, 20, 20), font=fonts['regular'])
    y_t += 18
    for char in CHARACTERS:
        clip_r  = clip_metrics['per_char_recall'].get(char, 0)
        vjepa_r = vjepa_metrics['per_char_recall'].get(char, 0)
        winner  = 'CLIP' if clip_r >= vjepa_r else 'V-JEPA 2'
        draw.text((PADDING, y_t),
                  f'{char:<22} {clip_r:>12.3f} {vjepa_r:>14.3f}   [{winner}]',
                  fill=(20, 20, 20), font=fonts['regular'])
        y_t += 18

    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(180, 180, 180), width=2)
    return canvas


def run():
    print('Loading manually labeled embeddings...')
    clip_emb, vjepa_emb, labels_true, char_to_int, panel_ids, metadata = load_labeled_embeddings()

    print(f'Labeled panels: {len(panel_ids)}')
    print(f'Distribution: {dict(Counter(labels_true))} -> {char_to_int}')

    print('\nRunning CLIP clustering...')
    clip_preds, clip_metrics = run_clustering(clip_emb, labels_true, char_to_int, 'CLIP')

    print('Running V-JEPA 2 clustering...')
    vjepa_preds, vjepa_metrics = run_clustering(vjepa_emb, labels_true, char_to_int, 'V-JEPA 2')

    print('\nResults:')
    print(f'{"Metric":<20} {"CLIP":>10} {"V-JEPA 2":>12}')
    print('-' * 44)
    for key in ['n_clusters', 'n_noise', 'purity', 'ari', 'nmi']:
        print(f'{key:<20} {clip_metrics[key]:>10} {vjepa_metrics[key]:>12}')

    print('\nPer-character recall:')
    print(f'{"Character":<22} {"CLIP":>8} {"V-JEPA 2":>10} {"Winner":>10}')
    print('-' * 52)
    for char in CHARACTERS:
        cr = clip_metrics['per_char_recall'].get(char, 0)
        vr = vjepa_metrics['per_char_recall'].get(char, 0)
        w  = 'CLIP' if cr >= vr else 'V-JEPA 2'
        print(f'{char:<22} {cr:>8.3f} {vr:>10.3f} {w:>10}')

    grid = build_grid(panel_ids, labels_true, clip_preds, vjepa_preds,
                      char_to_int, clip_metrics, vjepa_metrics)

    out_path = OUTPUT_DIR / 'oldboy_cluster_eval.jpg'
    grid.save(out_path, quality=95)
    print(f'\nGrid saved: {out_path}')
    print('Open with: eog data/oldboy_cluster_eval/oldboy_cluster_eval.jpg')


if __name__ == '__main__':
    run()