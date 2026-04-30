import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

PANELS_DIR     = Path(os.getenv('PANELS_DIR'))
EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
LABELS_PATH    = Path('data/labels/all_volumes_character_labels.json')
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

MANGA_VOLUMES = ['old_boy_vol01', 'old_boy_vol02', 'old_boy_vol03']
SAMPLE_SIZE   = 500

CHARACTERS = [
    'Shinichi Goto',
    'Takaaki Kakinuma',
    'Eri',
    'Multiple characters',
    'None / Background',
]


def load_labels() -> dict:
    if LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            return json.load(f)
    return {}


def save_labels(labels: dict):
    with open(LABELS_PATH, 'w') as f:
        json.dump(labels, f, indent=2)


def sample_panels(n: int = SAMPLE_SIZE) -> list[tuple[str, str]]:
    """
    Sample panels spread across all volumes using farthest point sampling.
    Returns list of (manga_name, panel_id) tuples.
    Skips already labeled panels.
    """
    all_embeddings = []
    all_panel_refs = []
    labels         = load_labels()

    for manga_name in MANGA_VOLUMES:
        emb_path  = EMBEDDINGS_DIR / manga_name / 'clip_embeddings.npy'
        meta_path = EMBEDDINGS_DIR / manga_name / 'metadata.json'
        if not emb_path.exists():
            print(f'Warning: embeddings not found for {manga_name}, skipping.')
            continue
        embeddings = np.load(emb_path)
        with open(meta_path) as f:
            metadata = json.load(f)
        for i, m in enumerate(metadata):
            ref = f'{manga_name}:{m["panel_id"]}'
            if ref not in labels:
                all_embeddings.append(embeddings[i])
                all_panel_refs.append((manga_name, m['panel_id']))

    print(f'Unlabeled panels available: {len(all_panel_refs)}')

    if len(all_panel_refs) <= n:
        return all_panel_refs

    all_embeddings = np.array(all_embeddings)
    norms          = np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8
    all_embeddings = all_embeddings / norms

    # Farthest point sampling for diverse coverage
    selected = [np.random.randint(len(all_panel_refs))]
    for _ in range(n - 1):
        dists = np.min(
            [np.sum((all_embeddings - all_embeddings[s]) ** 2, axis=1)
             for s in selected],
            axis=0
        )
        selected.append(int(np.argmax(dists)))

    return [all_panel_refs[i] for i in selected]


def get_panel_image(manga_name: str, panel_id: str) -> Image.Image:
    panel_path = PANELS_DIR / manga_name / f'{panel_id}.jpg'
    return Image.open(panel_path).convert('RGB')


def get_current(state):
    idx    = state['idx']
    panels = state['panels']
    if idx >= len(panels):
        labeled = len(state['labels'])
        return None, f'All done! {labeled} panels labeled across all volumes.', idx, len(panels)
    manga_name, panel_id = panels[idx]
    img      = get_panel_image(manga_name, panel_id)
    vol_num  = manga_name.replace('old_boy_vol0', 'Vol ')
    progress = (
        f'Panel {idx + 1} / {len(panels)}  |  '
        f'{vol_num} | {panel_id}  |  '
        f'Total labeled: {len(state["labels"])}'
    )
    return img, progress, idx, len(panels)


def save_and_next(character, state):
    idx    = state['idx']
    panels = state['panels']
    if idx >= len(panels):
        return state, None, 'All panels labeled!', gr.update()
    manga_name, panel_id   = panels[idx]
    ref                    = f'{manga_name}:{panel_id}'
    state['labels'][ref]   = character
    save_labels(state['labels'])
    state['idx']           = idx + 1
    img, progress, _, _    = get_current(state)
    return state, img, progress, gr.update(value=None)


def skip(state):
    idx    = state['idx']
    panels = state['panels']
    if idx < len(panels):
        manga_name, panel_id = panels[idx]
        ref                  = f'{manga_name}:{panel_id}'
        state['labels'][ref] = 'None / Background'
        save_labels(state['labels'])
    state['idx']        = idx + 1
    img, progress, _, _ = get_current(state)
    return state, img, progress, gr.update(value=None)


def undo(state):
    if state['idx'] > 0:
        state['idx'] -= 1
        manga_name, panel_id = state['panels'][state['idx']]
        ref = f'{manga_name}:{panel_id}'
        if ref in state['labels']:
            del state['labels'][ref]
            save_labels(state['labels'])
    img, progress, _, _ = get_current(state)
    return state, img, progress, gr.update(value=None)


def get_stats(state):
    from collections import Counter
    labels  = state['labels']
    counts  = Counter(labels.values())
    lines   = [f'{char}: {counts.get(char, 0)}' for char in CHARACTERS]

    # Per volume breakdown
    lines.append('\nPer volume:')
    for vol in MANGA_VOLUMES:
        vol_labels = {k: v for k, v in labels.items() if k.startswith(vol)}
        vol_counts = Counter(vol_labels.values())
        lines.append(f'  {vol}: {len(vol_labels)} labeled')
        for char in CHARACTERS[:3]:
            if vol_counts.get(char, 0) > 0:
                lines.append(f'    {char}: {vol_counts[char]}')

    lines.append(f'\nTotal labeled: {len(labels)} / {len(state["panels"])}')
    return '\n'.join(lines)


def build_app():
    sampled_panels = sample_panels(SAMPLE_SIZE)
    labels         = load_labels()
    state          = {'idx': 0, 'panels': sampled_panels, 'labels': labels}

    with gr.Blocks(title='MangaJEPA Panel Tagger v2') as app:
        gr.Markdown('# MangaJEPA Panel Tagger v2')
        gr.Markdown(
            f'Label **{SAMPLE_SIZE} panels** across **3 volumes** of Old Boy.  \n'
            f'Target: 15-20 labeled panels per character minimum.'
        )

        state_var = gr.State(state)

        with gr.Row():
            with gr.Column(scale=2):
                panel_img     = gr.Image(
                    label='Current Panel',
                    height=500,
                    interactive=False,
                )
                progress_text = gr.Markdown('Loading...')

            with gr.Column(scale=1):
                gr.Markdown('### Who is the main character?')
                character_radio = gr.Radio(
                    choices=CHARACTERS,
                    label='Character',
                    value=None,
                )

                with gr.Row():
                    save_btn = gr.Button('Save & Next', variant='primary')
                    skip_btn = gr.Button('Skip -> None/BG')
                    undo_btn = gr.Button('Undo')

                gr.Markdown('---')
                stats_btn  = gr.Button('Show Stats')
                stats_text = gr.Textbox(
                    label='Label Distribution',
                    interactive=False,
                    lines=18,
                )

        app.load(
            fn=lambda s: (get_current(s)[0], get_current(s)[1]),
            inputs=[state_var],
            outputs=[panel_img, progress_text],
        )

        save_btn.click(
            fn=save_and_next,
            inputs=[character_radio, state_var],
            outputs=[state_var, panel_img, progress_text, character_radio],
        )

        skip_btn.click(
            fn=skip,
            inputs=[state_var],
            outputs=[state_var, panel_img, progress_text, character_radio],
        )

        undo_btn.click(
            fn=undo,
            inputs=[state_var],
            outputs=[state_var, panel_img, progress_text, character_radio],
        )

        stats_btn.click(
            fn=get_stats,
            inputs=[state_var],
            outputs=[stats_text],
        )

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(server_port=7861, share=False)
