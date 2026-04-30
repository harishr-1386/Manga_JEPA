import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from src.store import get_collection
import open_clip

load_dotenv()

PANELS_DIR = Path(os.getenv('PANELS_DIR'))

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