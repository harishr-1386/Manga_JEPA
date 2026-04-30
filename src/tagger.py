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
MANGA_NAME     = 'old_boy_vol01'
LABELS_PATH    = Path(f'data/labels/{MANGA_NAME}_character_labels.json')
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Main cast
CHARACTERS = [
    'Shinichi Goto',
    'Takaaki Kakinuma',
    'Yukio Kusama',
    'Eri',
    'Multiple characters',
    #'None / Background',
]

# Sample size per session
SAMPLE_SIZE = 120


def load_labels() -> dict:
    if LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            return json.load(f)
    return {}


def save_labels(labels: dict):
    with open(LABELS_PATH, 'w') as f:
        json.dump(labels, f, indent=2)


def is_text_bubble(panel_id: str, white_threshold: float = 0.85) -> bool:
    """Return True if panel is likely a text bubble (mostly white pixels)."""
    try:
        img  = get_panel_image(panel_id)
        arr  = np.array(img.convert('L'))  # grayscale
        white_ratio = (arr > 240).sum() / arr.size
        return white_ratio > white_threshold
    except Exception:
        return False


def sample_panels(n: int = SAMPLE_SIZE) -> list[str]:
    """
    Sample panels spread across embedding space using farthest point sampling.
    This ensures diverse coverage rather than clustering near similar panels.
    """
    embeddings = np.load(EMBEDDINGS_DIR / MANGA_NAME / 'clip_embeddings.npy')
    with open(EMBEDDINGS_DIR / MANGA_NAME / 'metadata.json') as f:
        metadata = json.load(f)

    panel_ids = [m['panel_id'] for m in metadata]
    labels    = load_labels()

    # Prefer unlabeled panels
    unlabeled_idx = [i for i, pid in enumerate(panel_ids) if pid not in labels
    if not is_text_bubble(panel_ids[i])]

    if len(unlabeled_idx) <= n:
        return [panel_ids[i] for i in unlabeled_idx]

    # Farthest point sampling from unlabeled panels
    embs    = embeddings[unlabeled_idx]
    embs    = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    selected = [np.random.randint(len(unlabeled_idx))]

    for _ in range(n - 1):
        dists = np.min(
            [np.sum((embs - embs[s]) ** 2, axis=1) for s in selected],
            axis=0
        )
        selected.append(int(np.argmax(dists)))

    return [panel_ids[unlabeled_idx[i]] for i in selected]


def get_panel_image(panel_id: str) -> Image.Image:
    panel_path = PANELS_DIR / MANGA_NAME / f'{panel_id}.jpg'
    return Image.open(panel_path).convert('RGB')


def build_app():
    # Load session state
    sampled_panels = sample_panels(SAMPLE_SIZE)
    labels         = load_labels()
    state          = {'idx': 0, 'panels': sampled_panels, 'labels': labels}

    def get_current(state):
        idx      = state['idx']
        panels   = state['panels']
        if idx >= len(panels):
            return None, f'Done! {len(state["labels"])} panels labeled.', idx, len(panels)
        panel_id = panels[idx]
        img      = get_panel_image(panel_id)
        progress = f'Panel {idx + 1} / {len(panels)}  |  Total labeled: {len(state["labels"])}'
        return img, progress, idx, len(panels)

    def save_and_next(character, state):
        idx      = state['idx']
        panels   = state['panels']
        if idx >= len(panels):
            return state, None, 'All panels labeled!', gr.update()

        panel_id           = panels[idx]
        state['labels'][panel_id] = character
        save_labels(state['labels'])
        state['idx']       = idx + 1

        img, progress, _, _ = get_current(state)
        labeled_count = len(state['labels'])
        return state, img, progress, gr.update(value=None)

    # def skip(state):
    #     state['idx'] = state['idx'] + 1
    #     img, progress, _, _ = get_current(state)
    #     return state, img, progress, gr.update(value=None)

    def skip(state):
        idx      = state['idx']
        panels   = state['panels']
        if idx < len(panels):
            panel_id = panels[idx]
            state['labels'][panel_id] = 'None / Background'
            save_labels(state['labels'])
        state['idx'] = idx + 1
        img, progress, _, _ = get_current(state)
        return state, img, progress, gr.update(value=None)

    def undo(state):
        if state['idx'] > 0:
            state['idx'] -= 1
            prev_panel = state['panels'][state['idx']]
            if prev_panel in state['labels']:
                del state['labels'][prev_panel]
                save_labels(state['labels'])
        img, progress, _, _ = get_current(state)
        return state, img, progress, gr.update(value=None)

    def get_stats(state):
        labels  = state['labels']
        from collections import Counter
        counts  = Counter(labels.values())
        lines   = [f'{char}: {counts.get(char, 0)}' for char in CHARACTERS]
        lines.append(f'\nTotal labeled: {len(labels)}')
        return '\n'.join(lines)

    with gr.Blocks(title='MangaJEPA Panel Tagger', theme=gr.themes.Monochrome()) as app:
        gr.Markdown('# MangaJEPA — Panel Character Tagger')
        gr.Markdown(f'Label panels for **Old Boy Vol. 1**. Target: {SAMPLE_SIZE} panels.')

        state_var = gr.State(state)

        with gr.Row():
            with gr.Column(scale=2):
                panel_img = gr.Image(
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
                    skip_btn = gr.Button('Skip')
                    undo_btn = gr.Button('Undo')

                gr.Markdown('---')
                stats_btn  = gr.Button('Show Stats')
                stats_text = gr.Textbox(
                    label='Label Distribution',
                    interactive=False,
                    lines=8,
                )

        # Load first panel on startup
        init_img, init_progress, _, _ = get_current(state)

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
    app.launch(server_port=7860, share=False)