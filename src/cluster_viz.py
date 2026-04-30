import os
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from src.manga109_parser import parse_annotations, get_all_faces

load_dotenv()

EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
OUTPUT_DIR     = Path('data/cluster_viz')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Distinct colors for clusters — enough for 20+ clusters
CLUSTER_COLORS = [
    (220, 50,  50),   # red
    (50,  120, 220),  # blue
    (50,  180, 50),   # green
    (220, 160, 50),   # orange
    (160, 50,  220),  # purple
    (50,  200, 200),  # cyan
    (220, 50,  160),  # pink
    (100, 180, 80),   # lime
    (80,  80,  180),  # indigo
    (200, 120, 80),   # brown
    (50,  160, 160),  # teal
    (180, 80,  120),  # rose
    (120, 200, 160),  # mint
    (200, 200, 80),   # yellow
    (160, 100, 200),  # lavender
    (80,  160, 200),  # sky
    (200, 80,  80),   # salmon
    (80,  200, 120),  # sea green
    (200, 160, 200),  # lilac
    (160, 200, 80),   # yellow green
]
NOISE_COLOR = (60, 60, 60)  # dark gray for noise points


def get_color(label: int) -> tuple:
    if label == -1:
        return NOISE_COLOR
    return CLUSTER_COLORS[label % len(CLUSTER_COLORS)]


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


def crop_face(image_path: str, face: dict, size: int = 64, margin: float = 0.2):
    """Crop and resize a face to a fixed square thumbnail."""
    try:
        img  = Image.open(image_path).convert('RGB')
        w, h = img.size
        xmin, ymin = face['xmin'], face['ymin']
        xmax, ymax = face['xmax'], face['ymax']
        bw = xmax - xmin
        bh = ymax - ymin
        xmin = max(0, int(xmin - bw * margin))
        ymin = max(0, int(ymin - bh * margin))
        xmax = min(w, int(xmax + bw * margin))
        ymax = min(h, int(ymax + bh * margin))
        if (xmax - xmin) < 10 or (ymax - ymin) < 10:
            return None
        return img.crop((xmin, ymin, xmax, ymax)).resize((size, size))
    except Exception:
        return None


def build_cluster_panel(
    faces: list[dict],
    labels: np.ndarray,
    title_text: str,
    char_names: dict,
    labels_true: np.ndarray,
    face_size: int = 64,
    faces_per_row: int = 20,
    max_faces: int = 200,
) -> Image.Image:
    """
    Build a grid of face thumbnails colored by cluster label.
    Each face has a colored border indicating its predicted cluster.
    Ground truth character name shown below each face.
    """
    fonts      = get_fonts()
    border     = 3
    label_h    = 14
    cell_size  = face_size + border * 2
    cell_h     = cell_size 

    # Sample faces if too many
    indices = list(range(len(faces)))
    if len(indices) > max_faces:
        step    = len(indices) // max_faces
        indices = indices[::step][:max_faces]

    n_cols  = faces_per_row
    n_rows  = (len(indices) + n_cols - 1) // n_cols

    header_h = 50
    legend_h = 40
    total_w  = n_cols * cell_size + 20
    total_h  = header_h + n_rows * cell_h + legend_h + 10

    canvas = Image.new('RGB', (total_w, total_h), (245, 245, 245))
    draw   = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([0, 0, total_w, header_h], fill=(30, 30, 30))
    draw.text((10, 10), title_text, fill=(255, 255, 255), font=fonts['title'])

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    subtitle   = f'Clusters: {n_clusters}  |  Noise: {n_noise}  |  Total faces: {len(faces)}'
    draw.text((10, 30), subtitle, fill=(180, 180, 180), font=fonts['small'])

    # Face grid
    for grid_i, face_i in enumerate(indices):
        col = grid_i % n_cols
        row = grid_i // n_cols
        x   = 10 + col * cell_size
        y   = header_h + row * cell_h

        label      = labels[face_i]
        color      = get_color(label)
        face_data  = faces[face_i]

        # Colored border rectangle
        draw.rectangle(
            [x, y, x + cell_size - 1, y + cell_size - 1],
            fill=color
        )

        # Face thumbnail
        thumb = crop_face(face_data['image_path'], face_data, size=face_size)
        if thumb:
            canvas.paste(thumb, (x + border, y + border))
        else:
            draw.rectangle(
                [x + border, y + border,
                 x + cell_size - border, y + cell_size - border],
                fill=(200, 200, 200)
            )

        # Ground truth character name below face - to be removed - only japanese names are present - does not add to context 
        # true_label = labels_true[face_i]
        # char_id    = sorted(set(f['character_id'] for f in faces))[true_label]
        # char_name  = char_names.get(char_id, '?')[:6]
        # draw.text(
        #     (x, y + cell_size),
        #     char_name,
        #     fill=(40, 40, 40),
        #     font=fonts['small']
        # )

    # Legend
    y_legend = header_h + n_rows * cell_h + 5
    draw.text((10, y_legend), 'Border color = predicted cluster  |  Text = ground truth character', 
              fill=(100, 100, 100), font=fonts['small'])
    draw.rectangle([10, y_legend + 16, 24, y_legend + 30], fill=NOISE_COLOR)
    draw.text((28, y_legend + 16), '= noise (unassigned)', fill=(100, 100, 100), font=fonts['small'])

    return canvas


def visualize_title(title: str):
    """Generate comparison grid for one manga title."""
    char_dir = EMBEDDINGS_DIR / 'characters' / title

    if not char_dir.exists():
        print(f'No clustering data for {title} — run character_clusterer.py first')
        return

    # Load saved data
    labels_true  = np.load(char_dir / 'labels_true.npy')
    clip_labels  = np.load(char_dir / 'clip_labels_pred.npy')
    vjepa_labels = np.load(char_dir / 'vjepa_labels_pred.npy')

    with open(char_dir / 'metrics.json') as f:
        metrics = json.load(f)

    # Parse annotations to get face data and char names
    ann   = parse_annotations(title)
    faces = get_all_faces(ann)
    faces = [f for f in faces if f['character_id'] is not None]
    char_names = ann['characters']

    print(f'Generating visualization for {title}...')

    fonts = get_fonts()

    # Build three panels: ground truth, CLIP, V-JEPA 2
    gt_panel = build_cluster_panel(
        faces, labels_true,
        f'Ground Truth — {title}',
        char_names, labels_true,
    )
    clip_panel = build_cluster_panel(
        faces, clip_labels,
        f'CLIP Clustering — purity: {metrics["clip"]["purity"]:.4f} | ARI: {metrics["clip"]["ari"]:.4f}',
        char_names, labels_true,
    )
    vjepa_panel = build_cluster_panel(
        faces, vjepa_labels,
        f'V-JEPA 2 Clustering — purity: {metrics["vjepa"]["purity"]:.4f} | ARI: {metrics["vjepa"]["ari"]:.4f}',
        char_names, labels_true,
    )

    # Stack three panels vertically
    total_w = max(gt_panel.width, clip_panel.width, vjepa_panel.width)
    total_h = gt_panel.height + clip_panel.height + vjepa_panel.height + 20

    final = Image.new('RGB', (total_w, total_h), (200, 200, 200))
    final.paste(gt_panel,    (0, 0))
    final.paste(clip_panel,  (0, gt_panel.height + 5))
    final.paste(vjepa_panel, (0, gt_panel.height + clip_panel.height + 10))

    out_path = OUTPUT_DIR / f'{title}_cluster_comparison.jpg'
    final.save(out_path, quality=95)
    print(f'Saved: {out_path}')
    return out_path


if __name__ == '__main__':
    titles = ['ARMS', 'AisazuNihaIrarenai', 'Akuhamu']
    for title in titles:
        visualize_title(title)

    print('\nOpen results:')
    print('eog data/cluster_viz/')