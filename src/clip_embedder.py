import os
import json
import numpy as np
import open_clip
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb

load_dotenv()

PANELS_DIR = Path(os.getenv('PANELS_DIR'))
EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
CHROMA_DIR = Path(os.getenv('EMBEDDINGS_DIR')).parent / 'chroma'

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model.eval()


def encode_panels_clip(manga_name: str, batch_size: int = 32):
    panel_dir = PANELS_DIR / manga_name
    panel_files = sorted(panel_dir.glob('*.jpg'))

    with open(EMBEDDINGS_DIR / manga_name / 'metadata.json') as f:
        metadata = json.load(f)

    print(f'CLIP encoding {len(panel_files)} panels...')

    all_embeddings = []
    for i in tqdm(range(0, len(panel_files), batch_size)):
        batch_files = panel_files[i:i + batch_size]
        imgs = torch.stack([
            preprocess(Image.open(f).convert('RGB'))
            for f in batch_files
        ])
        with torch.no_grad():
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Save to disk
    np.save(EMBEDDINGS_DIR / manga_name / 'clip_embeddings.npy', embeddings)
    print(f'CLIP embeddings shape: {embeddings.shape}')

    # Load into ChromaDB as separate collection
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = f'{manga_name}_clip'

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'}
    )

    batch_size = 100
    for i in range(0, len(metadata), batch_size):
        collection.add(
            embeddings=embeddings[i:i + batch_size].tolist(),
            metadatas=metadata[i:i + batch_size],
            ids=[m['panel_id'] for m in metadata[i:i + batch_size]]
        )

    print(f'Loaded {collection.count()} panels into "{collection_name}"')
    return embeddings, metadata


if __name__ == '__main__':
    encode_panels_clip('old_boy_vol01')