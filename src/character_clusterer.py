import os
import json
import sys
import numpy as np
import torch
import open_clip
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from src.manga109_parser import parse_annotations, get_all_faces
import importlib.util
import os
from src.encoder import load_encoder

load_dotenv()


# def _load_vjepa_vit():
#     from dotenv import load_dotenv
#     load_dotenv()
#     vjepa_path = os.getenv('VJEPA2_HUB_PATH')
#     spec = importlib.util.spec_from_file_location(
#         'vision_transformer',
#         os.path.join(vjepa_path, 'src', 'models', 'vision_transformer.py')
#     )
#     mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(mod)
#     return mod.vit_large

# vit_large = _load_vjepa_vit()



MANGA109_IMAGES = Path(os.getenv('MANGA109_IMAGES'))
OUTPUT_DIR      = Path(os.getenv('EMBEDDINGS_DIR')) / 'characters'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load CLIP once at module level
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model.eval()


def crop_face(image_path: str, face: dict, margin: float = 0.2):
    """
    Crop a face from a page image with optional margin.
    Returns None if crop is too small to be useful.
    """
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

    if (xmax - xmin) < 20 or (ymax - ymin) < 20:
        return None

    return img.crop((xmin, ymin, xmax, ymax))


def embed_faces_clip(faces: list[dict], batch_size: int = 32):
    """
    Embed all face crops using CLIP image encoder.
    Returns embeddings array and list of valid indices.
    """
    embeddings   = []
    valid_indices = []
    crops        = []

    for i, face in enumerate(faces):
        crop = crop_face(face['image_path'], face)
        if crop is not None:
            crops.append((i, clip_preprocess(crop)))

    print(f'Valid face crops: {len(crops)} / {len(faces)}')

    for batch_start in tqdm(range(0, len(crops), batch_size), desc='CLIP face embedding'):
        batch          = crops[batch_start:batch_start + batch_size]
        indices, tensors = zip(*batch)
        imgs           = torch.stack(tensors)
        with torch.no_grad():
            feats = clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
        valid_indices.extend(indices)

    return np.concatenate(embeddings, axis=0), valid_indices


def embed_faces_vjepa(faces: list[dict], valid_idx: list[int], device: str = 'cuda') -> np.ndarray:
    """
    Embed face crops using V-JEPA 2 encoder.
    Uses the same valid_idx from CLIP to keep comparison fair.
    """
    from torchvision import transforms
    #sys.path.insert(0, os.getenv('VJEPA2_HUB_PATH'))
    #from src.models.vision_transformer import vit_large

    weights_path = os.getenv('VJEPA2_WEIGHTS')
    # model = _vit_large(
    #     img_size=256,
    #     patch_size=16,
    #     num_frames=8,
    #     tubelet_size=2,
    #     uniform_power=True,
    #     use_sdpa=True,
    # )
    with tqdm(total=1, desc='  [4/5] Loading V-JEPA 2', unit='step') as pbar:
        model = load_encoder(device)
        pbar.update(1)


    # ckpt    = torch.load(weights_path, map_location='cpu')
    # cleaned = {
    #     k.replace('module.backbone.', ''): v
    #     for k, v in ckpt['encoder'].items()
    # }
    # model.load_state_dict(cleaned, strict=False)
    # model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    embeddings  = []
    valid_faces = [faces[i] for i in valid_idx]

    for face in tqdm(valid_faces, desc='V-JEPA 2 face embedding'):
        crop = crop_face(face['image_path'], face)
        if crop is None:
            embeddings.append(np.zeros(1024))
            continue
        frame = transform(crop)
        clip  = frame.unsqueeze(1).repeat(1, 8, 1, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(clip).mean(dim=1).cpu().numpy()[0]
        embeddings.append(emb)

    return np.array(embeddings)


def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Cluster purity: for each predicted cluster, what fraction
    of its members share the majority true label?
    """
    from collections import Counter
    total  = len(labels_true)
    purity = 0.0
    for cluster_id in set(labels_pred):
        if cluster_id == -1:
            continue
        mask                  = labels_pred == cluster_id
        true_labels_in_cluster = labels_true[mask]
        most_common_count     = Counter(true_labels_in_cluster).most_common(1)[0][1]
        purity               += most_common_count
    return round(purity / total, 4)


def run_hdbscan(embeddings: np.ndarray) -> np.ndarray:
    clusterer = HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric='euclidean',
        copy=True,
    )
    return clusterer.fit_predict(normalize(embeddings))


def compute_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    mask = labels_pred != -1
    if mask.sum() < 5:
        return {'n_clusters': 0, 'n_noise': int((labels_pred == -1).sum()),
                'purity': 0.0, 'ari': 0.0, 'nmi': 0.0}

    n_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    n_noise    = int((labels_pred == -1).sum())
    purity     = cluster_purity(labels_true[mask], labels_pred[mask])
    ari        = adjusted_rand_score(labels_true[mask], labels_pred[mask])
    nmi        = normalized_mutual_info_score(labels_true[mask], labels_pred[mask])

    return {
        'n_clusters': n_clusters,
        'n_noise':    n_noise,
        'purity':     purity,
        'ari':        round(float(ari), 4),
        'nmi':        round(float(nmi), 4),
    }


def evaluate_clustering(title: str) -> dict:
    """
    Full pipeline:
    1. Parse Manga109 annotations for a title
    2. Embed all faces with CLIP and V-JEPA 2
    3. Cluster each embedding set with HDBSCAN
    4. Evaluate both against ground truth character IDs
    """
    print(f'\nProcessing: {title}')
    ann   = parse_annotations(title)
    faces = get_all_faces(ann)
    faces = [f for f in faces if f['character_id'] is not None]
    print(f'Faces with character ID: {len(faces)}')

    if len(faces) < 10:
        print('Too few faces, skipping.')
        return None

    # Ground truth integer labels
    char_ids    = sorted(set(f['character_id'] for f in faces))
    char_to_int = {c: i for i, c in enumerate(char_ids)}
    labels_true = np.array([char_to_int[f['character_id']] for f in faces])
    n_true_chars = len(char_ids)

    # CLIP embeddings + clustering
    print('Embedding faces with CLIP...')
    clip_embeddings, valid_idx = embed_faces_clip(faces)
    labels_true_valid          = labels_true[valid_idx]

    print('Clustering CLIP embeddings...')
    clip_labels  = run_hdbscan(clip_embeddings)
    clip_metrics = compute_metrics(labels_true_valid, clip_labels)

    # V-JEPA 2 embeddings + clustering
    print('Embedding faces with V-JEPA 2...')
    device         = 'cuda' if torch.cuda.is_available() else 'cpu'
    vjepa_embeddings = embed_faces_vjepa(faces, valid_idx, device)

    print('Clustering V-JEPA 2 embeddings...')
    vjepa_labels  = run_hdbscan(vjepa_embeddings)
    vjepa_metrics = compute_metrics(labels_true_valid, vjepa_labels)

    result = {
        'title':        title,
        'n_faces':      len(valid_idx),
        'n_true_chars': n_true_chars,
        'clip':         clip_metrics,
        'vjepa':        vjepa_metrics,
    }

    # Print comparison table
    print(f'\nResults for {title}:')
    print(f'  {"Metric":<12} {"CLIP":>10} {"V-JEPA 2":>10}')
    print(f'  {"-"*34}')
    print(f'  {"Clusters":<12} {clip_metrics["n_clusters"]:>10} {vjepa_metrics["n_clusters"]:>10}')
    print(f'  {"Noise":<12} {clip_metrics["n_noise"]:>10} {vjepa_metrics["n_noise"]:>10}')
    print(f'  {"Purity":<12} {clip_metrics["purity"]:>10.4f} {vjepa_metrics["purity"]:>10.4f}')
    print(f'  {"ARI":<12} {clip_metrics["ari"]:>10.4f} {vjepa_metrics["ari"]:>10.4f}')
    print(f'  {"NMI":<12} {clip_metrics["nmi"]:>10.4f} {vjepa_metrics["nmi"]:>10.4f}')

    # Save to disk
    out = OUTPUT_DIR / title
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'clip_face_embeddings.npy',  clip_embeddings)
    np.save(out / 'vjepa_face_embeddings.npy', vjepa_embeddings)
    np.save(out / 'labels_true.npy',           labels_true_valid)
    np.save(out / 'clip_labels_pred.npy',      clip_labels)
    np.save(out / 'vjepa_labels_pred.npy',     vjepa_labels)
    with open(out / 'metrics.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result


def print_divider(char='=', width=72):
    print(char * width)


if __name__ == '__main__':
    test_titles = ['ARMS', 'AisazuNihaIrarenai', 'Akuhamu']
    all_results = []

    for title in test_titles:
        result = evaluate_clustering(title)
        if result:
            all_results.append(result)

    print_divider()
    print('CLUSTERING SUMMARY: CLIP vs V-JEPA 2')
    print_divider()
    print(f'{"Title":<25} {"Chars":>6} | '
          f'{"CLIP Purity":>11} {"CLIP ARI":>9} {"CLIP NMI":>9} | '
          f'{"VJEPA Purity":>12} {"VJEPA ARI":>9} {"VJEPA NMI":>9}')
    print('-' * 72)

    for r in all_results:
        print(f'{r["title"]:<25} {r["n_true_chars"]:>6} | '
              f'{r["clip"]["purity"]:>11.4f} {r["clip"]["ari"]:>9.4f} {r["clip"]["nmi"]:>9.4f} | '
              f'{r["vjepa"]["purity"]:>12.4f} {r["vjepa"]["ari"]:>9.4f} {r["vjepa"]["nmi"]:>9.4f}')

    if all_results:
        def avg(enc, key):
            return round(
                sum(r[enc][key] for r in all_results) / len(all_results), 4
            )

        print('-' * 72)
        print(f'{"AVERAGE":<25} {"":>6} | '
              f'{avg("clip","purity"):>11.4f} {avg("clip","ari"):>9.4f} {avg("clip","nmi"):>9.4f} | '
              f'{avg("vjepa","purity"):>12.4f} {avg("vjepa","ari"):>9.4f} {avg("vjepa","nmi"):>9.4f}')

    print_divider()