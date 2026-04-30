import os
import json
import numpy as np
import chromadb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
CHROMA_DIR = Path(os.getenv('EMBEDDINGS_DIR')).parent / 'chroma'


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def load_manga_into_chroma(manga_name: str):
    """Load precomputed embeddings + metadata into ChromaDB."""
    client = get_client()

    # Drop and recreate collection for clean reload
    try:
        client.delete_collection(manga_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=manga_name,
        metadata={"hnsw:space": "cosine"}
    )

    emb_path = EMBEDDINGS_DIR / manga_name
    embeddings = np.load(emb_path / 'embeddings.npy')
    with open(emb_path / 'metadata.json') as f:
        metadata = json.load(f)

    print(f'Loading {len(metadata)} panels into ChromaDB...')

    # ChromaDB has a batch size limit — insert in chunks
    batch_size = 100
    for i in range(0, len(metadata), batch_size):
        batch_emb = embeddings[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]

        collection.add(
            embeddings=batch_emb.tolist(),
            metadatas=batch_meta,
            ids=[m['panel_id'] for m in batch_meta]
        )

    print(f'Done. Collection "{manga_name}" has {collection.count()} panels.')
    return collection


def get_collection(manga_name: str):
    client = get_client()
    return client.get_collection(manga_name)


if __name__ == '__main__':
    collection = load_manga_into_chroma('old_boy_vol01')
    print(f'\nCollection count: {collection.count()}')

    # Sanity check — query with the first embedding
    embeddings = np.load(
        EMBEDDINGS_DIR / 'old_boy_vol01' / 'embeddings.npy'
    )
    results = collection.query(
        query_embeddings=[embeddings[0].tolist()],
        n_results=3
    )
    print('\nTop 3 similar panels to panel 0:')
    for meta in results['metadatas'][0]:
        print(f"  page {meta['page']:04d} panel {meta['panel']:04d}")