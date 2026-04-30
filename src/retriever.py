import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from src.store import get_collection
import open_clip

load_dotenv()

PANELS_DIR_STR = os.getenv('PANELS_DIR')
if not PANELS_DIR_STR:
    raise EnvironmentError('PANELS_DIR not set. Copy .env.example to .env and fill in your paths.')
PANELS_DIR = Path(PANELS_DIR_STR)

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.eval()


def encode_text_query(query: str) -> np.ndarray:
    tokens = clip_tokenizer([query])
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()[0]


def retrieve_panels(
    manga_name: str,
    query: str,
    n_results: int = 5,
) -> list[dict]:
    collection = get_collection(f'{manga_name}_clip')
    query_embedding = encode_text_query(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
    )

    panels = []
    for meta, distance in zip(
        results['metadatas'][0],
        results['distances'][0]
    ):
        panels.append({
            **meta,
            'similarity': 1 - distance,
        })

    return panels



# def retrieve_panels_for_character(
#     manga_name: str,
#     character: str,
#     query: str,
#     n_results: int = 5,
# ) -> list[dict]:
#     """
#     Character-aware retrieval:
#     1. Load registry to find panels containing the character
#     2. Score those panels against the query
#     3. Return top-k matches
#     """
#     import json
#     from pathlib import Path
# # remove - cos i added in label propagated data -- 
#     # registry_path = Path(f'data/labels/{manga_name}_character_registry.json')
#     # if not registry_path.exists():
#     #     # Fall back to standard retrieval if no registry
#     #     return retrieve_panels(manga_name, query, n_results)
# # till here 
    
#     propagated_path = Path(f'data/labels/{manga_name}_propagated_registry.json')
#     manual_path     = Path(f'data/labels/{manga_name}_character_registry.json')
#     registry_path   = propagated_path if propagated_path.exists() else manual_path
#     if not registry_path.exists():
#         # Fall back to standard retrieval if no registry
#         return retrieve_panels(manga_name, query, n_results)


# Intersection Retrieval to get a balance between - best semantic match and character based match.
def retrieve_panels_for_character(
    manga_name: str,
    character: str,
    query: str,
    n_results: int = 5,
    broad_k: int = 100,
) -> list[dict]:
    """
    Intersection retrieval:
    1. Broad semantic retrieval (top broad_k panels)
    2. Filter to panels in character registry
    3. Return top-n from filtered set
    Combines semantic relevance with character identity.
    """
    import json
    from pathlib import Path

    # Load registry
    propagated_path = Path(f'data/labels/{manga_name}_propagated_registry.json')
    manual_path     = Path(f'data/labels/{manga_name}_character_registry.json')
    registry_path   = propagated_path if propagated_path.exists() else manual_path

    if not registry_path.exists():
        return retrieve_panels(manga_name, query, n_results)

    with open(registry_path) as f:
        registry = json.load(f)

    # Character panel set
    char_panel_ids = {
        pid for pid, char in registry.items()
        if char is not None
        and isinstance(char, str)
        and char.lower() == character.lower()
    }

    if not char_panel_ids:
        print(f'No labeled panels for {character}, falling back to standard retrieval')
        return retrieve_panels(manga_name, query, n_results)

    # Broad semantic retrieval
    broad_results = retrieve_panels(manga_name, query, n_results=broad_k)

    # Filter to character panels
    filtered = [r for r in broad_results if r['panel_id'] in char_panel_ids]

    if len(filtered) >= n_results:
        return filtered[:n_results]

    # Fallback: pad with remaining broad results if not enough character panels
    seen     = {r['panel_id'] for r in filtered}
    fallback = [r for r in broad_results if r['panel_id'] not in seen]
    return (filtered + fallback)[:n_results]

    
    
    
    with open(registry_path) as f:
        registry = json.load(f)

    # # Get panel IDs for this character
    # character_panels = [
    #     pid for pid, char in registry.items()
    #     if char.lower() == character.lower()
    # ]

    # Get panel IDs for this character
# Guard against None or non-string values in registry
    character_panels = [
        pid for pid, char in registry.items()
        if char is not None
        and isinstance(char, str)
        and char.lower() == character.lower()
    ]

    if not character_panels:
        print(f'No labeled panels found for {character}, falling back to standard retrieval')
        return retrieve_panels(manga_name, query, n_results)

    # Load CLIP embeddings and metadata
    embeddings_dir = Path(os.getenv('EMBEDDINGS_DIR'))
    clip_embeddings = np.load(embeddings_dir / manga_name / 'clip_embeddings.npy')

    import json as _json
    with open(embeddings_dir / manga_name / 'metadata.json') as f:
        metadata = _json.load(f)

    panel_id_to_idx = {m['panel_id']: i for i, m in enumerate(metadata)}

    # Get embeddings for character panels only
    char_indices = [
        panel_id_to_idx[pid]
        for pid in character_panels
        if pid in panel_id_to_idx
    ]

    if not char_indices:
        return retrieve_panels(manga_name, query, n_results)

    # Encode query
    query_embedding = encode_text_query(query)

    # Score character panels against query
    char_embeddings = clip_embeddings[char_indices]
    char_embeddings_norm = char_embeddings / (
        np.linalg.norm(char_embeddings, axis=1, keepdims=True) + 1e-8
    )
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    scores = char_embeddings_norm @ query_norm

    # Sort by score
    sorted_order = np.argsort(scores)[::-1][:n_results]

    results = []
    for rank_idx in sorted_order:
        orig_idx  = char_indices[rank_idx]
        meta      = metadata[orig_idx]
        results.append({
            **meta,
            'similarity':  float(scores[rank_idx]),
            'character':   character,
        })

    return results


if __name__ == '__main__':
    queries = [
        "a man eating food",
        "two people fighting",
        "a person looking scared",
        "a dark alley at night",
    ]

    for query in queries:
        print(f'\nQuery: "{query}"')
        results = retrieve_panels('old_boy_vol01', query, n_results=3)
        for r in results:
            print(f"  page {r['page']:04d} panel {r['panel']:04d} "
                  f"(similarity: {r['similarity']:.3f}) → {r['path']}")