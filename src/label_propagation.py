import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
MANGA_NAME     = 'old_boy_vol01'
LABELS_PATH    = Path(f'data/labels/{MANGA_NAME}_character_labels.json')
REGISTRY_PATH  = Path(f'data/labels/{MANGA_NAME}_character_registry.json')
PROPAGATED_PATH = Path(f'data/labels/{MANGA_NAME}_propagated_registry.json')


def propagate_labels(
    similarity_threshold: float = 0.82,
    max_neighbors: int = 10,
) -> dict:
    """
    For each labeled character panel, find nearest neighbors
    in CLIP embedding space and propagate the label if similarity
    is above threshold.
    """
    # Load embeddings and metadata
    clip_embeddings = np.load(EMBEDDINGS_DIR / MANGA_NAME / 'clip_embeddings.npy')
    with open(EMBEDDINGS_DIR / MANGA_NAME / 'metadata.json') as f:
        metadata = json.load(f)

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    panel_ids       = [m['panel_id'] for m in metadata]
    panel_id_to_idx = {pid: i for i, pid in enumerate(panel_ids)}

    # Normalize embeddings for cosine similarity
    norms           = np.linalg.norm(clip_embeddings, axis=1, keepdims=True) + 1e-8
    clip_norm       = clip_embeddings / norms

    propagated      = dict(registry)  # start with manual labels
    vote_counts     = defaultdict(lambda: defaultdict(float))

    print(f'Starting with {len(registry)} manually labeled panels')
    print(f'Propagating to neighbors (threshold={similarity_threshold})...')

    for panel_id, character in registry.items():
        if panel_id not in panel_id_to_idx:
            continue

        idx        = panel_id_to_idx[panel_id]
        query_emb  = clip_norm[idx]

        # Cosine similarity to all panels
        sims       = clip_norm @ query_emb

        # Get top neighbors excluding self
        top_indices = np.argsort(sims)[::-1]
        neighbors_added = 0

        for neighbor_idx in top_indices:
            if neighbor_idx == idx:
                continue
            if sims[neighbor_idx] < similarity_threshold:
                break
            if neighbors_added >= max_neighbors:
                break

            neighbor_id = panel_ids[neighbor_idx]
            # Accumulate weighted votes per panel
            vote_counts[neighbor_id][character] += float(sims[neighbor_idx])
            neighbors_added += 1

    # Assign label based on highest weighted vote
    newly_labeled = 0
    for panel_id, votes in vote_counts.items():
        if panel_id not in propagated:
            best_char = max(votes, key=votes.get)
            propagated[panel_id] = best_char
            newly_labeled += 1

    print(f'Propagated labels to {newly_labeled} new panels')
    print(f'Total registry size: {len(propagated)}')

    # Distribution
    from collections import Counter
    dist = Counter(propagated.values())
    print('\nPropagated label distribution:')
    for char, count in dist.most_common():
        print(f'  {char}: {count}')

    # Save
    with open(PROPAGATED_PATH, 'w') as f:
        json.dump(propagated, f, indent=2)
    print(f'\nSaved to {PROPAGATED_PATH}')

    return propagated


if __name__ == '__main__':
    propagated = propagate_labels(
        similarity_threshold=0.82,
        max_neighbors=10,
    )