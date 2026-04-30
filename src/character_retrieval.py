import os
import json
import numpy as np
import chromadb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
CHROMA_DIR     = Path(os.getenv('EMBEDDINGS_DIR')).parent / 'chroma'
PANELS_DIR     = Path(os.getenv('PANELS_DIR'))

VOLUMES    = ['old_boy_vol01', 'old_boy_vol02', 'old_boy_vol03']
CHARACTERS = ['Shinichi Goto', 'Takaaki Kakinuma', 'Eri']

MULTI_LABELS_PATH  = Path('data/labels/all_volumes_character_labels.json')
SINGLE_LABELS_PATH = Path('data/labels/old_boy_vol01_character_registry.json')
INDEX_PATH         = Path('data/labels/character_index.json')


def load_all_embeddings(volumes: list[str]) -> tuple[dict, dict]:
    all_emb  = {}
    all_meta = {}
    for vol in volumes:
        emb_path = EMBEDDINGS_DIR / vol / 'embeddings.npy'
        if not emb_path.exists():
            print(f'Warning: embeddings not found for {vol}, skipping.')
            continue
        all_emb[vol] = np.load(emb_path)
        with open(EMBEDDINGS_DIR / vol / 'metadata.json') as f:
            meta = json.load(f)
        all_meta[vol] = {m['panel_id']: i for i, m in enumerate(meta)}
    return all_emb, all_meta


def build_prototype(
    character: str,
    all_labels: dict,
    all_emb: dict,
    all_meta: dict,
    use_multi: bool,
) -> tuple[np.ndarray, int]:
    """Build a character prototype vector from seed panels."""
    seed_vecs = []
    for ref, char in all_labels.items():
        if char != character:
            continue
        if use_multi:
            if ':' not in ref:
                continue
            manga_name, panel_id = ref.split(':', 1)
        else:
            manga_name = VOLUMES[0]
            panel_id   = ref

        if manga_name not in all_meta:
            continue
        if panel_id not in all_meta[manga_name]:
            continue
        idx = all_meta[manga_name][panel_id]
        seed_vecs.append(all_emb[manga_name][idx])

    if not seed_vecs:
        return None, 0

    proto = np.mean(seed_vecs, axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    return proto, len(seed_vecs)


def orthogonalize_prototype(
    target: np.ndarray,
    others: list[np.ndarray],
    strength: float = 0.3,
) -> np.ndarray:
    """
    Remove other characters signatures from target prototype.
    Reduces cross-character contamination.
    """
    refined = target.copy()
    for other in others:
        projection = np.dot(refined, other) * other
        refined    = refined - strength * projection
    refined = refined / (np.linalg.norm(refined) + 1e-8)
    return refined


def query_all_volumes(
    prototype: np.ndarray,
    volumes: list[str],
    top_k: int,
    character: str,
) -> list[dict]:
    client  = chromadb.PersistentClient(path=str(CHROMA_DIR))
    results = []
    for vol in volumes:
        try:
            col = client.get_collection(vol)
            res = col.query(
                query_embeddings=[prototype.tolist()],
                n_results=min(top_k, col.count()),
            )
            for meta, dist in zip(res['metadatas'][0], res['distances'][0]):
                results.append({
                    **meta,
                    'similarity': round(1 - dist, 4),
                    'character':  character,
                    'volume':     vol,
                })
        except Exception as e:
            print(f'  Warning: {vol} query failed: {e}')
    results.sort(key=lambda r: r['similarity'], reverse=True)
    return results


def build_character_index(
    characters: list[str] = None,
    volumes: list[str] = None,
    top_k: int = 200,
    ortho_strength: float = 0.3,
    save: bool = True,
) -> dict:
    """
    Build character panel index using V-JEPA 2 prototype retrieval.
    Applies prototype orthogonalization and Goto-panel filtering
    to reduce cross-character contamination.
    """
    if characters is None:
        characters = CHARACTERS
    if volumes is None:
        volumes = VOLUMES

    # Load labels
    if MULTI_LABELS_PATH.exists():
        with open(MULTI_LABELS_PATH) as f:
            all_labels = json.load(f)
        use_multi = True
        print(f'Using multi-volume labels: {len(all_labels)} entries')
    else:
        with open(SINGLE_LABELS_PATH) as f:
            all_labels = json.load(f)
        use_multi = False
        print(f'Using single-volume labels: {len(all_labels)} entries')

    # Load embeddings
    print('Loading V-JEPA 2 embeddings...')
    all_emb, all_meta = load_all_embeddings(volumes)

    # Build raw prototypes
    print('\nBuilding prototypes...')
    raw_prototypes = {}
    for character in characters:
        proto, n_seeds = build_prototype(
            character, all_labels, all_emb, all_meta, use_multi
        )
        if proto is None:
            print(f'  {character}: no seeds found, skipping')
            continue
        raw_prototypes[character] = proto
        print(f'  {character}: {n_seeds} seeds')

    # Orthogonalize prototypes
    print('\nOrthogonalizing prototypes...')
    refined_prototypes = {}
    for character, proto in raw_prototypes.items():
        others  = [p for c, p in raw_prototypes.items() if c != character]
        refined = orthogonalize_prototype(proto, others, strength=ortho_strength)
        shift   = 1 - float(np.dot(proto, refined))
        refined_prototypes[character] = refined
        print(f'  {character}: shift = {shift:.4f}')

    # Retrieve panels + filter
    print('\nRetrieving panels...')
    index          = {}
    goto_panel_ids = set()

    # Process Goto first to build exclusion set
    ordered = ['Shinichi Goto'] + [c for c in characters if c != 'Shinichi Goto']

    for character in ordered:
        if character not in refined_prototypes:
            continue

        results = query_all_volumes(
            refined_prototypes[character], volumes, top_k * 2, character
        )

        if character == 'Shinichi Goto':
            # Build Goto exclusion set from top results
            goto_panel_ids = {
                f'{r["volume"]}:{r["panel_id"]}'
                for r in results[:top_k]
            }
            index[character] = results[:top_k]
        else:
            # Filter Goto panels from other characters
            before   = len(results)
            filtered = [
                r for r in results
                if f'{r["volume"]}:{r["panel_id"]}' not in goto_panel_ids
            ]
            removed  = before - len(filtered)
            index[character] = filtered[:top_k]
            if removed > 0:
                print(f'  {character}: removed {removed} Goto-contaminated panels')

        vol_counts = {}
        for r in index[character]:
            vol_counts[r['volume']] = vol_counts.get(r['volume'], 0) + 1
        sim_range = (
            f'{index[character][-1]["similarity"]:.3f}'
            f' - {index[character][0]["similarity"]:.3f}'
        ) if index[character] else 'N/A'
        print(f'  {character}: {len(index[character])} panels {vol_counts} | sim: {sim_range}')

    if save:
        with open(INDEX_PATH, 'w') as f:
            json.dump(index, f, indent=2)
        print(f'\nCharacter index saved to {INDEX_PATH}')

    return index


def get_character_panels(
    character: str,
    top_k: int = 200,
) -> list[dict]:
    """
    Load character panels from saved index.
    Adds path field for panel images.
    """
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            index = json.load(f)
        if character in index:
            panels = index[character][:top_k]
            for p in panels:
                p['path'] = str(
                    PANELS_DIR / p['volume'] / f'{p["panel_id"]}.jpg'
                )
            return panels

    print(f'No index found, computing on the fly for {character}')
    return build_character_index([character])[character]


if __name__ == '__main__':
    print('Building character index with orthogonalization + Goto filtering')
    print('=' * 65)
    index = build_character_index()

    print('\n' + '=' * 65)
    print('FINAL CHARACTER INDEX SUMMARY')
    print('=' * 65)
    for char, panels in index.items():
        vol_counts = {}
        for p in panels:
            vol_counts[p['volume']] = vol_counts.get(p['volume'], 0) + 1
        print(f'\n{char}:')
        print(f'  Total: {len(panels)}')
        for vol, count in sorted(vol_counts.items()):
            print(f'  {vol}: {count}')
        if panels:
            print(f'  Sim: {panels[-1]["similarity"]:.3f} - {panels[0]["similarity"]:.3f}')
