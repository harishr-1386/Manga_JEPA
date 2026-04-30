import os
import json
import numpy as np
import chromadb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
CHROMA_DIR     = Path(os.getenv('EMBEDDINGS_DIR')).parent / 'chroma'
REGISTRY_PATH  = Path('data/labels/old_boy_vol01_character_registry.json')

VOLUMES = ['old_boy_vol01', 'old_boy_vol02', 'old_boy_vol03']


def get_seed_embedding(character: str, manga_name: str = 'old_boy_vol01') -> np.ndarray:
    """
    Average V-JEPA 2 embeddings of all manually labeled panels
    for a character to produce one character prototype vector.
    """
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    vjepa_emb = np.load(EMBEDDINGS_DIR / manga_name / 'embeddings.npy')
    with open(EMBEDDINGS_DIR / manga_name / 'metadata.json') as f:
        metadata = json.load(f)

    panel_id_to_idx = {m['panel_id']: i for i, m in enumerate(metadata)}

    seed_vectors = []
    for panel_id, char in registry.items():
        if char != character:
            continue
        if panel_id not in panel_id_to_idx:
            continue
        seed_vectors.append(vjepa_emb[panel_id_to_idx[panel_id]])

    if not seed_vectors:
        raise ValueError(f'No labeled panels found for character: {character}')

    prototype = np.mean(seed_vectors, axis=0)
    prototype = prototype / (np.linalg.norm(prototype) + 1e-8)

    print(f'Character prototype for {character}: mean of {len(seed_vectors)} seed panels')
    return prototype


def find_character_panels(
    character: str,
    volumes: list[str] = None,
    top_k: int = 200,
    min_similarity: float = 0.0,
) -> list[dict]:
    """
    Find all panels visually similar to a character's prototype
    across all specified volumes using V-JEPA 2 embeddings.

    Returns panels ranked by visual similarity to the character.
    """
    if volumes is None:
        volumes = VOLUMES

    prototype = get_seed_embedding(character)
    client    = chromadb.PersistentClient(path=str(CHROMA_DIR))

    all_results = []

    for manga_name in volumes:
        try:
            collection = client.get_collection(manga_name)
        except Exception:
            print(f'Collection {manga_name} not found, skipping.')
            continue

        results = collection.query(
            query_embeddings=[prototype.tolist()],
            n_results=min(top_k, collection.count()),
        )

        for meta, distance in zip(
            results['metadatas'][0],
            results['distances'][0],
        ):
            similarity = 1 - distance
            if similarity < min_similarity:
                continue
            all_results.append({
                **meta,
                'similarity': round(similarity, 4),
                'character':  character,
                'volume':     manga_name,
            })

    # Sort across all volumes by similarity
    all_results.sort(key=lambda r: r['similarity'], reverse=True)
    return all_results[:top_k]


def build_character_index(
    characters: list[str],
    volumes: list[str] = None,
    top_k: int = 200,
    save: bool = True,
) -> dict:
    """
    Build a complete character index across all volumes.
    Returns {character: [panel_dicts ranked by similarity]}
    Optionally saves to disk.
    """
    if volumes is None:
        volumes = VOLUMES

    index = {}
    for character in characters:
        print(f'\nFinding panels for: {character}')
        panels = find_character_panels(character, volumes, top_k)
        index[character] = panels
        print(f'  Found {len(panels)} panels across {len(volumes)} volumes')

        vol_counts = {}
        for p in panels:
            vol_counts[p['volume']] = vol_counts.get(p['volume'], 0) + 1
        for vol, count in vol_counts.items():
            print(f'    {vol}: {count} panels')

        if panels:
            print(f'  Similarity range: {panels[-1]["similarity"]:.3f} - {panels[0]["similarity"]:.3f}')

    if save:
        out_path = Path('data/labels/character_index.json')
        # Save only serializable fields
        serializable = {
            char: [
                {k: v for k, v in p.items() if k != 'path'}
                for p in panels
            ]
            for char, panels in index.items()
        }
        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f'\nCharacter index saved to {out_path}')

    return index


def get_character_panels(
    character: str,
    top_k: int = 200,
) -> list[dict]:
    """
    Load character panels from saved index if available,
    otherwise compute on the fly.
    """
    index_path = Path('data/labels/character_index.json')
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        if character in index:
            panels = index[character]
            # Re-add path field
            for p in panels:
                panels_dir = Path(os.getenv('PANELS_DIR'))
                p['path'] = str(panels_dir / p['manga'] / f'{p["panel_id"]}.jpg')
            return panels[:top_k]

    return find_character_panels(character, top_k=top_k)


if __name__ == '__main__':
    CHARACTERS = ['Shinichi Goto', 'Takaaki Kakinuma', 'Eri']

    print('Building character index using V-JEPA 2 prototype retrieval...')
    print('=' * 60)

    index = build_character_index(CHARACTERS, top_k=200)

    print('\n' + '=' * 60)
    print('CHARACTER INDEX SUMMARY')
    print('=' * 60)
    for char, panels in index.items():
        vol_counts = {}
        for p in panels:
            vol_counts[p['volume']] = vol_counts.get(p['volume'], 0) + 1
        print(f'\n{char}:')
        print(f'  Total panels: {len(panels)}')
        for vol, count in sorted(vol_counts.items()):
            print(f'  {vol}: {count}')
        if panels:
            print(f'  Top similarity: {panels[0]["similarity"]:.3f}')
            print(f'  Min similarity: {panels[-1]["similarity"]:.3f}')
