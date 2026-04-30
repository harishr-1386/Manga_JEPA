import os
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from src.retriever import retrieve_panels, retrieve_panels_for_character

load_dotenv()

PANELS_DIR = Path(os.getenv('PANELS_DIR'))
OUTPUT_DIR = Path('data/method_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANGA_NAME  = 'old_boy_vol01'
PANEL_SZ    = 160
PADDING     = 12
N_RESULTS   = 5

QUERIES = [
    'two people fighting',
    'a person looking sad or defeated',
    'a dark threatening scene',
]

CHARACTERS = ['Shinichi Goto', 'Takaaki Kakinuma', 'Eri']


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


def load_panel(path: str, size: int = PANEL_SZ) -> Image.Image:
    try:
        img   = Image.open(path).convert('RGB')
        ratio = size / max(img.width, img.height)
        img   = img.resize((int(img.width * ratio), int(img.height * ratio)))
        canvas = Image.new('RGB', (size, size), (220, 220, 220))
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        return canvas
    except Exception:
        return Image.new('RGB', (size, size), (200, 200, 200))


def old_character_retrieval(manga_name, character, query, n_results=5):
    """
    Simulate old approach: search only within labeled registry panels.
    Uses direct embedding lookup without broad semantic search.
    """
    embeddings_dir = Path(os.getenv('EMBEDDINGS_DIR'))
    clip_embeddings = np.load(embeddings_dir / manga_name / 'clip_embeddings.npy')
    with open(embeddings_dir / manga_name / 'metadata.json') as f:
        metadata = json.load(f)

    propagated_path = Path(f'data/labels/{manga_name}_propagated_registry.json')
    with open(propagated_path) as f:
        registry = json.load(f)

    panel_ids       = [m['panel_id'] for m in metadata]
    panel_id_to_idx = {pid: i for i, pid in enumerate(panel_ids)}

    char_indices = [
        panel_id_to_idx[pid]
        for pid, char in registry.items()
        if char == character and pid in panel_id_to_idx
    ]

    if not char_indices:
        return []

    from src.retriever import encode_text_query
    query_emb  = encode_text_query(query)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

    char_embs  = clip_embeddings[char_indices]
    char_norms = char_embs / (np.linalg.norm(char_embs, axis=1, keepdims=True) + 1e-8)
    scores     = char_norms @ query_norm

    top_idx = np.argsort(scores)[::-1][:n_results]
    results = []
    for i in top_idx:
        orig_idx = char_indices[i]
        meta     = metadata[orig_idx]
        results.append({**meta, 'similarity': float(scores[i])})
    return results


def build_three_way_grid(query: str, character: str) -> Image.Image:
    fonts = get_fonts()

    std_results  = retrieve_panels(manga_name=MANGA_NAME, query=query, n_results=N_RESULTS)
    old_results  = old_character_retrieval(MANGA_NAME, character, query, N_RESULTS)
    new_results  = retrieve_panels_for_character(MANGA_NAME, character, query, N_RESULTS)

    METHODS = [
        ('Standard Retrieval',      std_results,  (40,  80,  160)),
        ('Old Character-Aware',     old_results,  (180, 60,  60)),
        ('Intersection Retrieval',  new_results,  (40,  140, 80)),
    ]

    HEADER_H   = 55
    METHOD_H   = 30
    SUBLABEL_H = 18
    METRIC_H   = 45
    total_w    = PADDING + N_RESULTS * (PANEL_SZ + PADDING)
    total_w    = max(total_w, 950)
    section_h  = METHOD_H + PANEL_SZ + SUBLABEL_H + METRIC_H
    total_h    = HEADER_H + len(METHODS) * (section_h + PADDING) + PADDING

    canvas = Image.new('RGB', (total_w, total_h), (248, 248, 248))
    draw   = ImageDraw.Draw(canvas)

    # Query header
    draw.rectangle([0, 0, total_w, HEADER_H], fill=(20, 20, 20))
    draw.text((PADDING, 10), f'Query: "{query}"', fill=(255, 255, 255), font=fonts['title'])
    draw.text((PADDING, 34), f'Character filter: {character}  |  n={N_RESULTS}',
              fill=(160, 160, 160), font=fonts['small'])

    y = HEADER_H + PADDING

    for method_name, results, color in METHODS:
        # Method header bar
        draw.rectangle([0, y, total_w, y + METHOD_H], fill=color)
        avg_sim = np.mean([r['similarity'] for r in results]) if results else 0
        draw.text(
            (PADDING, y + 8),
            f'{method_name}   avg sim: {avg_sim:.3f}',
            fill=(255, 255, 255), font=fonts['bold']
        )

        # Panels
        x = PADDING
        for r in results:
            img = load_panel(r['path'])
            canvas.paste(img, (x, y + METHOD_H))

            # Similarity badge
            draw.rectangle([x, y + METHOD_H, x + 48, y + METHOD_H + 16], fill=(0, 0, 0))
            draw.text(
                (x + 2, y + METHOD_H + 2),
                f'{r["similarity"]:.3f}',
                fill=(255, 220, 0), font=fonts['small']
            )

            # Page label
            label = f'p{r["page"]:04d}|pan{r["panel"]:02d}'
            draw.text(
                (x, y + METHOD_H + PANEL_SZ + 2),
                label, fill=(80, 80, 80), font=fonts['small']
            )
            x += PANEL_SZ + PADDING

        # Metrics row
        y_metric = y + METHOD_H + PANEL_SZ + SUBLABEL_H
        draw.rectangle([0, y_metric, total_w, y_metric + METRIC_H], fill=(235, 235, 240))

        sims     = [r['similarity'] for r in results]
        max_sim  = max(sims) if sims else 0
        min_sim  = min(sims) if sims else 0

        draw.text(
            (PADDING, y_metric + 6),
            f'avg: {avg_sim:.3f}   max: {max_sim:.3f}   min: {min_sim:.3f}   panels in pool: {"all 1250" if method_name == "Standard Retrieval" else f"{character} registry only" if method_name == "Old Character-Aware" else "semantic top-100 intersect registry"}',
            fill=(40, 40, 40), font=fonts['small']
        )

        y += section_h + PADDING

    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(180, 180, 180), width=2)
    return canvas


if __name__ == '__main__':
    for character in CHARACTERS:
        char_slug = character.replace(' ', '_')
        char_grids = []

        for query in QUERIES:
            print(f'Building: {character} | {query}')
            grid = build_three_way_grid(query, character)
            char_grids.append(grid)

        # Stack vertically
        total_h = sum(g.height for g in char_grids) + len(char_grids) * 8
        total_w = max(g.width for g in char_grids)
        stacked = Image.new('RGB', (total_w, total_h), (200, 200, 200))
        y = 0
        for g in char_grids:
            stacked.paste(g, (0, y))
            y += g.height + 8

        out_path = OUTPUT_DIR / f'{char_slug}_method_comparison.jpg'
        stacked.save(out_path, quality=95)
        print(f'Saved: {out_path}')

    print('\nOpen with: eog data/method_comparison/')