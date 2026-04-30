import os
import json
import base64
import ollama
import numpy as np
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
from src.retriever import retrieve_panels, retrieve_panels_for_character

load_dotenv()


_missing = [v for v in ['PANELS_DIR', 'EMBEDDINGS_DIR']
            if not os.getenv(v)]
if _missing:
    raise EnvironmentError(
        f'Missing env vars: {_missing}\n'
        'Copy .env.example to .env and fill in your paths:\n'
        'cp .env.example .env'
    )

PANELS_DIR  = Path(os.getenv('PANELS_DIR'))
OUTPUT_DIR  = Path('data/action_labels')
INDEX_PATH  = Path('data/labels/character_index.json')

# PANELS_DIR    = Path(os.getenv('PANELS_DIR'))
# OUTPUT_DIR    = Path('data/action_labels')
# INDEX_PATH    = Path('data/labels/character_index.json')

VOLUMES    = ['old_boy_vol01', 'old_boy_vol02', 'old_boy_vol03']
CHARACTERS = ['All', 'Shinichi Goto', 'Takaaki Kakinuma', 'Eri']

ACTION_SHORT = [
    'All', 'fighting', 'running', 'talking', 'sad', 'angry',
    'eating', 'dark scene', 'walking alone', 'flashback',
    'revelation', 'captured', 'crowd',
]

VOL_MAP = {
    'All':   None,
    'Vol 1': 'old_boy_vol01',
    'Vol 2': 'old_boy_vol02',
    'Vol 3': 'old_boy_vol03',
}


# --- Helpers ---

def load_action_labels(volumes=None):
    if volumes is None:
        volumes = VOLUMES
    all_labels = {}
    for vol in volumes:
        path = OUTPUT_DIR / f'{vol}_clip_actions.json'
        if path.exists():
            with open(path) as f:
                all_labels.update(json.load(f))
    return all_labels


def load_character_index():
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            return json.load(f)
    return {}


def panel_to_image(panel_meta: dict) -> str:
    """Return image path for a panel."""
    vol = panel_meta.get('volume', panel_meta.get('manga', VOLUMES[0]))
    pid = panel_meta.get('panel_id')
    return str(PANELS_DIR / vol / f'{pid}.jpg')


def encode_image(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def filter_by_action(panels: list[dict], action: str, all_labels: dict) -> list[dict]:
    if action == 'All':
        return panels
    return [
        p for p in panels
        if all_labels.get(p.get('panel_id', ''), {}).get('action') == action
    ]


def filter_by_volume(panels: list[dict], volume_key: str) -> list[dict]:
    vol = VOL_MAP.get(volume_key)
    if vol is None:
        return panels
    return [
        p for p in panels
        if p.get('volume', p.get('manga', '')) == vol
    ]


def build_prompt(query: str, panels: list[dict], all_labels: dict) -> str:
    panel_descs = []
    for i, p in enumerate(panels):
        pid    = p.get('panel_id', '')
        vol    = p.get('volume', p.get('manga', ''))
        page   = p.get('page', '')
        action = all_labels.get(pid, {}).get('action', 'unknown')
        char   = p.get('character', '')
        char_str = f', character: {char}' if char else ''
        panel_descs.append(
            f'Panel {i+1}: {vol} page {page:04d}{char_str}, action: {action}'
        )

    panels_context = '\n'.join(panel_descs)

    return f"""You are a manga analyst for Old Boy (Dark Horse edition).
You have been given {len(panels)} manga panels retrieved as visual evidence.

Panel context:
{panels_context}

QUESTION: {query}

STRICT RULES:
- Answer ONLY based on what you can see in the provided panel images
- Do NOT use prior knowledge about Old Boy from your training data
- Reference specific panels in your answer (e.g. Panel 1, Panel 3)
- If panels lack enough information, say so explicitly
- Be concise and specific

Answer based purely on the visual evidence:"""


# def get_llm_response(
#     query: str,
#     panels: list[dict],
#     all_labels: dict,
#     model_choice: str,
#     api_key: str = '',
# ) -> str:
#     prompt  = build_prompt(query, panels, all_labels)
#     images  = []
#     for p in panels:
#         path = panel_to_image(p)
#         if Path(path).exists():
#             images.append(encode_image(path))

#     if model_choice == 'Gemini Flash' and api_key.strip():
#         try:
#             from google import genai
#             from google.genai import types
#             client   = genai.Client(api_key=api_key.strip())
#             content  = [prompt]
#             for img_b64 in images:
#                 content.append(types.Part.from_bytes(
#                     data=base64.b64decode(img_b64),
#                     mime_type='image/jpeg'
#                 ))
#             response = client.models.generate_content(
#                 model='gemini-2.0-flash', contents=content
#             )
#             return response.text
#         except Exception as e:
#             return f'Gemini error: {e}\nFalling back to Ollama...\n\n' + \
#                    get_llm_response(query, panels, all_labels, 'Ollama (local)', '')

#     else:
#         # Ollama — Llama 3.2 Vision with one image at a time
#         try:
#             messages = []
#             for i, (img_b64, p) in enumerate(zip(images, panels)):
#                 messages.append({
#                     'role':    'user',
#                     'content': f'Panel {i+1}',
#                     'images':  [img_b64],
#                 })
#                 messages.append({
#                     'role':    'assistant',
#                     'content': f'I can see panel {i+1}.',
#                 })
#             messages.append({'role': 'user', 'content': prompt})
#             response = ollama.chat(model='llama3.2-vision', messages=messages)
#             return response['message']['content']
#         except Exception as e:
#             return f'Ollama error: {e}\nMake sure Ollama is running with llama3.2-vision.'


def get_llm_response(
    query: str,
    panels: list[dict],
    all_labels: dict,
    model_choice: str,
    api_key: str = '',
) -> str:
    prompt  = build_prompt(query, panels, all_labels)
    images  = []
    for p in panels:
        path = panel_to_image(p)
        if Path(path).exists():
            images.append(encode_image(path))

    if model_choice == 'Gemini Flash':
        if not api_key.strip():
            return 'Please enter a Gemini API key in the LLM Settings panel above.'
        try:
            from google import genai
            from google.genai import types
            client   = genai.Client(api_key=api_key.strip())
            content  = [prompt]
            for img_b64 in images:
                content.append(types.Part.from_bytes(
                    data=base64.b64decode(img_b64),
                    mime_type='image/jpeg'
                ))
            response = client.models.generate_content(
                model='gemini-2.0-flash', contents=content
            )
            return response.text
        except Exception as e:
            return f'Gemini error: {e}'

    else:
        # Ollama
        try:
            import ollama as ollama_client
            messages = []
            for i, (img_b64, p) in enumerate(zip(images, panels)):
                messages.append({
                    'role':    'user',
                    'content': f'Panel {i+1}',
                    'images':  [img_b64],
                })
                messages.append({
                    'role':    'assistant',
                    'content': f'I can see panel {i+1}.',
                })
            messages.append({'role': 'user', 'content': prompt})
            response = ollama_client.chat(model='llama3.2-vision', messages=messages)
            return response['message']['content']
        except ImportError:
            return (
                'Ollama Python client not installed. Run: pip install ollama\n\n'
                'Or switch to Gemini Flash and enter an API key above.'
            )
        except Exception as e:
            if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                return (
                    'Ollama is not running. Install and start it:\n\n'
                    'curl -fsSL https://ollama.com/install.sh | sh\n'
                    'ollama pull llama3.2-vision\n\n'
                    'Or switch to Gemini Flash and enter an API key above.'
                )
            return f'Ollama error: {e}'

# --- Tab 1: Q&A ---

def qa_query(
    query: str,
    character: str,
    action: str,
    volume: str,
    n_results: int,
    model_choice: str,
    api_key: str,
):
    if not query.strip():
        return [], 'Please enter a query.'

    all_labels = load_action_labels()

    # Retrieve panels
    if character != 'All':
        panels = retrieve_panels_for_character(
            'old_boy_vol01', character, query, n_results=50
        )
        # Supplement with other volumes from character index
        char_index = load_character_index()
        if character in char_index:
            extra = [
                p for p in char_index[character]
                if p.get('volume') != 'old_boy_vol01'
            ]
            panels = panels + extra[:30]
    else:
        panels = []
        for vol in VOLUMES:
            panels += retrieve_panels(vol, query, n_results=20)
        panels.sort(key=lambda r: r.get('similarity', 0), reverse=True)

    # Apply filters
    panels = filter_by_action(panels, action, all_labels)
    panels = filter_by_volume(panels, volume)
    panels = panels[:n_results]

    if not panels:
        return [], 'No panels found matching your filters. Try relaxing the filters.'

    # Build gallery
    gallery = []
    for p in panels:
        path = panel_to_image(p)
        if Path(path).exists():
            pid    = p.get('panel_id', '')
            vol    = p.get('volume', p.get('manga', ''))
            sim    = p.get('similarity', 0)
            action_label = all_labels.get(pid, {}).get('action', '')
            caption = f'{vol} | sim:{sim:.3f} | {action_label}'
            gallery.append((path, caption))

    # Get LLM response
    answer = get_llm_response(query, panels, all_labels, model_choice, api_key)

    return gallery, answer


# --- Tab 2: Character Explorer ---

def character_explorer(character: str, volume: str, top_k: int):
    if character == 'All':
        return [], 'Please select a specific character.'

    char_index = load_character_index()
    if character not in char_index:
        return [], f'No index found for {character}. Run character_retrieval.py first.'

    panels = char_index[character]
    panels = filter_by_volume(panels, volume)
    panels = panels[:top_k]

    gallery = []
    for p in panels:
        path = panel_to_image(p)
        if Path(path).exists():
            vol  = p.get('volume', '')
            sim  = p.get('similarity', 0)
            pid  = p.get('panel_id', '')
            caption = f'{vol} | {pid} | sim:{sim:.3f}'
            gallery.append((path, caption))

    vol_counts = {}
    for p in char_index[character]:
        v = p.get('volume', '')
        vol_counts[v] = vol_counts.get(v, 0) + 1

    stats = f'**{character}** — {len(char_index[character])} total panels\n\n'
    for vol, count in sorted(vol_counts.items()):
        stats += f'- {vol}: {count} panels\n'

    return gallery, stats


# --- Tab 3: Action Browser ---

def action_browser(action: str, volume: str, top_k: int):
    if action == 'All':
        return [], 'Please select a specific action.'

    all_labels = load_action_labels()

    matched = [
        v for v in all_labels.values()
        if v.get('action') == action
    ]
    matched.sort(key=lambda x: x.get('confidence', 0), reverse=True)

    # Filter by volume
    if VOL_MAP.get(volume):
        vol_name = VOL_MAP[volume]
        matched  = [m for m in matched if m.get('manga', '') == vol_name]

    matched = matched[:top_k]

    gallery = []
    for m in matched:
        pid  = m.get('panel_id', '')
        vol  = m.get('manga', VOLUMES[0])
        conf = m.get('confidence', 0)
        path = str(PANELS_DIR / vol / f'{pid}.jpg')
        if Path(path).exists():
            caption = f'{vol} | {pid} | conf:{conf:.3f}'
            gallery.append((path, caption))

    return gallery, f'Showing top {len(gallery)} panels for action: **{action}**'


# --- Build App ---

def build_app():
    with gr.Blocks(title='MangaJEPA') as app:

        gr.Markdown('# MangaJEPA')
        gr.Markdown(
            'Manga comprehension system using V-JEPA 2 + CLIP dual-encoder retrieval. '
            'Old Boy Vol. 1-3 (Dark Horse).'
        )

        # LLM settings in a collapsible row at top
        with gr.Accordion('LLM Settings', open=False):
            with gr.Row():
                model_choice = gr.Radio(
                    choices=['Ollama (local)', 'Gemini Flash'],
                    value='Gemini Flash',
                    label='LLM Backend',
                )
                api_key = gr.Textbox(
                    label='Gemini API Key (only needed for Gemini Flash)',
                    placeholder='AIza...',
                    type='password',
                )

        with gr.Tabs():

            # --- Tab 1: Q&A ---
            with gr.TabItem('Ask About the Manga'):
                gr.Markdown('Ask anything about the manga. Results are grounded in retrieved panels.')

                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label='Your Question',
                            placeholder='What is Goto doing in the dark scenes?',
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        qa_btn = gr.Button('Ask', variant='primary', scale=1)

                with gr.Row():
                    char_filter   = gr.Dropdown(
                        choices=CHARACTERS,
                        value='All',
                        label='Character Filter',
                    )
                    action_filter = gr.Dropdown(
                        choices=ACTION_SHORT,
                        value='All',
                        label='Action Filter',
                    )
                    vol_filter    = gr.Dropdown(
                        choices=['All', 'Vol 1', 'Vol 2', 'Vol 3'],
                        value='All',
                        label='Volume',
                    )
                    n_panels      = gr.Slider(
                        minimum=3, maximum=10, value=5, step=1,
                        label='Number of Panels',
                    )

                qa_gallery = gr.Gallery(
                    label='Retrieved Panels',
                    columns=5,
                    height=300,
                    object_fit='contain',
                )
                qa_answer  = gr.Markdown(label='Grounded Response')

                qa_btn.click(
                    fn=qa_query,
                    inputs=[
                        query_input, char_filter, action_filter,
                        vol_filter, n_panels, model_choice, api_key,
                    ],
                    outputs=[qa_gallery, qa_answer],
                )

            # --- Tab 2: Character Explorer ---
            with gr.TabItem('Character Explorer'):
                gr.Markdown('Browse panels by character using V-JEPA 2 prototype retrieval.')

                with gr.Row():
                    char_select  = gr.Dropdown(
                        choices=CHARACTERS[1:],
                        value='Shinichi Goto',
                        label='Character',
                    )
                    vol_select   = gr.Dropdown(
                        choices=['All', 'Vol 1', 'Vol 2', 'Vol 3'],
                        value='All',
                        label='Volume',
                    )
                    topk_slider  = gr.Slider(
                        minimum=5, maximum=50, value=20, step=5,
                        label='Number of Panels',
                    )
                    explore_btn  = gr.Button('Explore', variant='primary')

                char_gallery = gr.Gallery(
                    label='Character Panels',
                    columns=5,
                    height=400,
                    object_fit='contain',
                )
                char_stats   = gr.Markdown()

                explore_btn.click(
                    fn=character_explorer,
                    inputs=[char_select, vol_select, topk_slider],
                    outputs=[char_gallery, char_stats],
                )

            # --- Tab 3: Action Browser ---
            with gr.TabItem('Action Browser'):
                gr.Markdown('Browse panels by detected action (CLIP zero-shot classification).')

                with gr.Row():
                    action_select = gr.Dropdown(
                        choices=ACTION_SHORT[1:],
                        value='angry',
                        label='Action',
                    )
                    action_vol    = gr.Dropdown(
                        choices=['All', 'Vol 1', 'Vol 2', 'Vol 3'],
                        value='All',
                        label='Volume',
                    )
                    action_topk   = gr.Slider(
                        minimum=5, maximum=50, value=20, step=5,
                        label='Number of Panels',
                    )
                    browse_btn    = gr.Button('Browse', variant='primary')

                action_gallery = gr.Gallery(
                    label='Panels',
                    columns=5,
                    height=400,
                    object_fit='contain',
                )
                action_info    = gr.Markdown()

                browse_btn.click(
                    fn=action_browser,
                    inputs=[action_select, action_vol, action_topk],
                    outputs=[action_gallery, action_info],
                )

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(server_port=7860, share=False)
