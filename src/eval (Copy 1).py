import time
import base64
import ollama
from tqdm import tqdm
from src.retriever import retrieve_panels
from PIL import Image, ImageDraw, ImageFont
import os

MODELS = ['llama3.2-vision', 'moondream']

TEST_QUERIES = [
    "what is happening in the fighting scenes?",
    "describe the mood and atmosphere of the dark scenes",
    "what are characters doing in the eating scenes?",
]


def build_prompt(query: str, panels: list[dict]) -> str:
    return f"""You are a manga analyst. You have been given {len(panels)} manga panels
retrieved as visual evidence relevant to the following question:

QUESTION: {query}

STRICT RULES:
- Answer ONLY based on what you can see in the provided panels
- Do NOT use any prior knowledge about this manga from your training data
- If the panels don't contain enough information to answer, say so explicitly
- Reference specific panels in your answer (e.g. "In page 43 panel 1...")
- Be concise and specific

Answer the question based purely on the visual evidence provided."""


def panel_reference_rate(answer: str, panels: list[dict]) -> float:
    referenced = 0
    for panel in panels:
        page_str = str(panel['page'])
        panel_str = str(panel['panel'])
        if page_str in answer or f"panel {panel_str}" in answer.lower():
            referenced += 1
    return round(referenced / len(panels), 2)


def run_model(model: str, query: str, panels: list[dict]) -> dict:
    images = []
    for panel in panels:
        with open(panel['path'], 'rb') as f:
            images.append(base64.b64encode(f.read()).decode('utf-8'))

    start = time.time()

    if model == 'llama3.2-vision':
        # Llama only supports one image per message
        # Send each panel separately then ask the final question
        messages = []
        for i, (panel, img) in enumerate(zip(panels, images)):
            messages.append({
                'role': 'user',
                'content': f"Panel {i+1}: page {panel['page']}, panel {panel['panel']}",
                'images': [img],
            })
            messages.append({
                'role': 'assistant',
                'content': f'I can see panel {i+1}.',
            })
        messages.append({
            'role': 'user',
            'content': build_prompt(query, panels),
        })
        response = ollama.chat(model=model, messages=messages)

    else:
        # Moondream supports multiple images
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': build_prompt(query, panels),
                'images': images,
            }]
        )

    total_latency = time.time() - start
    answer = response['message']['content']

    input_tokens = response.get('prompt_eval_count', 0)
    output_tokens = response.get('eval_count', 0)
    eval_duration_s = response.get('eval_duration', 0) / 1e9
    prompt_duration_s = response.get('prompt_eval_duration', 0) / 1e9

    tokens_per_second = (
        round(output_tokens / eval_duration_s, 2)
        if eval_duration_s > 0 else 0
    )

    return {
        'model': model,
        'query': query,
        'answer': answer,
        'total_latency_s': round(total_latency, 2),
        'prompt_eval_time_s': round(prompt_duration_s, 2),
        'generation_time_s': round(eval_duration_s, 2),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'tokens_per_second': tokens_per_second,
        'input_output_ratio': round(input_tokens / output_tokens, 2) if output_tokens > 0 else 0,
        'panel_reference_rate': panel_reference_rate(answer, panels),
    }


def print_divider(char='=', width=70):
    print(char * width)

import textwrap 

# def save_panel_grid(panels: list[dict], query: str, model: str, out_dir: str = 'data/eval_grids'):
#     """Create a side-by-side grid of retrieved panels with labels."""
#     os.makedirs(out_dir, exist_ok=True)

#     images = [Image.open(p['path']).convert('RGB') for p in panels]

#     # Resize all panels to same height
#     target_h = 300
#     resized = []
#     for img in images:
#         ratio = target_h / img.height
#         new_w = int(img.width * ratio)
#         resized.append(img.resize((new_w, target_h)))

#     label_h = 30
#     total_w = sum(img.width for img in resized)
#     grid = Image.new('RGB', (total_w, target_h + label_h), color=(240, 240, 240))

#     try:
#         font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
#     except Exception:
#         font = ImageFont.load_default()

#     draw = ImageDraw.Draw(grid)
#     x_offset = 0
#     for img, panel in zip(resized, panels):
#         grid.paste(img, (x_offset, label_h))
#         label = f"p{panel['page']:04d} | pan{panel['panel']:02d} | {panel['similarity']:.2f}"
#         draw.text((x_offset + 4, 4), label, fill=(30, 30, 30), font=font)
#         x_offset += img.width

#     # Clean filename
#     query_slug = query[:30].replace(' ', '_').replace('?', '')
#     model_slug = model.replace(':', '-')
#     filename = f'{out_dir}/{model_slug}__{query_slug}.jpg'
#     grid.save(filename)
#     return filename


def save_panel_grid(panels: list[dict], query: str, model: str, answer: str, out_dir: str = 'data/eval_grids'):
    os.makedirs(out_dir, exist_ok=True)

    images = [Image.open(p['path']).convert('RGB') for p in panels]

    try:
        font_regular = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
        font_bold    = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
        font_title   = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
        font_caption = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
        font_small   = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except Exception:
        font_regular = font_bold = font_title = font_caption = font_small = ImageFont.load_default()

    # Layout constants
    PANEL_H      = 380
    LABEL_H      = 20   # panel letter label above image
    SUBLABEL_H   = 32   # similarity + page info below image
    PADDING      = 16
    GAP          = 6    # gap between panels
    BG           = (255, 255, 255)
    BORDER       = (180, 180, 180)
    TEXT_DARK    = (20,  20,  20 )
    TEXT_MED     = (80,  80,  80 )
    TEXT_ACCENT  = (30,  80,  160)  # academic blue
    TABLE_HEAD   = (240, 240, 240)

    # Resize panels to uniform height
    resized = []
    for img in images:
        ratio = PANEL_H / img.height
        resized.append(img.resize((max(int(img.width * ratio), 80), PANEL_H)))

    n = len(resized)
    panel_area_w = sum(img.width for img in resized) + GAP * (n - 1)

    # Metrics table height
    table_h = 22 * 3 + PADDING * 2  # 3 rows

    # Answer text wrapping
    answer_w = max(panel_area_w + PADDING * 2, 900)
    chars_per_line = (answer_w - PADDING * 2) // 7
    wrapped = textwrap.fill(answer, width=chars_per_line)
    answer_lines = wrapped.split('\n')
    line_h = 17
    answer_h = len(answer_lines) * line_h + PADDING * 2

    # Caption height
    caption_h = 40

    # Total canvas
    total_w = max(panel_area_w + PADDING * 2, 900)
    total_h = (PADDING + LABEL_H + PANEL_H + SUBLABEL_H + PADDING +
               table_h + PADDING + answer_h + caption_h + PADDING)

    canvas = Image.new('RGB', (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # ── Panel row ─────────────────────────────────────────────────────
    x = PADDING
    y_label    = PADDING
    y_img      = y_label + LABEL_H
    y_sublabel = y_img + PANEL_H

    for i, (img, panel) in enumerate(zip(resized, panels)):
        letter = f'({chr(97 + i)})'  # (a), (b), (c)...

        # Panel letter centered above image
        draw.text(
            (x + img.width // 2 - 8, y_label),
            letter, fill=TEXT_ACCENT, font=font_bold
        )

        # Panel image with thin border
        canvas.paste(img, (x, y_img))
        draw.rectangle(
            [x - 1, y_img - 1, x + img.width, y_img + PANEL_H],
            outline=BORDER, width=1
        )

        # Sublabel: page/panel + similarity score
        sub1 = f'Page {panel["page"]:04d}, Panel {panel["panel"]:02d}'
        sub2 = f'Similarity: {panel["similarity"]:.3f}'
        draw.text((x, y_sublabel + 2),  sub1, fill=TEXT_MED,  font=font_small)
        draw.text((x, y_sublabel + 16), sub2, fill=TEXT_DARK, font=font_bold)

        x += img.width + GAP

    # ── Metrics table ─────────────────────────────────────────────────
    y_table = y_sublabel + SUBLABEL_H + PADDING
    col_w   = total_w // 4
    headers = ['Model', 'Avg Similarity', 'Panel Ref Rate', 'Latency']
    avg_sim = round(sum(p['similarity'] for p in panels) / len(panels), 3)
    values  = [
        model,
        str(avg_sim),
        '(see summary)',
        '—',
    ]

    # Table header row
    draw.rectangle([PADDING, y_table, total_w - PADDING, y_table + 22], fill=TABLE_HEAD)
    draw.rectangle([PADDING, y_table, total_w - PADDING, y_table + 22], outline=BORDER, width=1)
    for col_i, h in enumerate(headers):
        draw.text((PADDING + col_i * col_w + 6, y_table + 4), h, fill=TEXT_DARK, font=font_bold)

    # Table value row
    row_y = y_table + 22
    draw.rectangle([PADDING, row_y, total_w - PADDING, row_y + 22], outline=BORDER, width=1)
    for col_i, v in enumerate(values):
        draw.text((PADDING + col_i * col_w + 6, row_y + 4), v, fill=TEXT_DARK, font=font_regular)

    # ── Answer text box ───────────────────────────────────────────────
    y_answer = y_table + table_h
    draw.rectangle([PADDING, y_answer, total_w - PADDING, y_answer + answer_h],
                   fill=(250, 250, 252), outline=BORDER, width=1)

    draw.text(
        (PADDING + 8, y_answer + PADDING - 2),
        f'Model Response ({model}):',
        fill=TEXT_ACCENT, font=font_title
    )

    y_text = y_answer + PADDING + 18
    for line in answer_lines:
        draw.text((PADDING + 8, y_text), line, fill=TEXT_DARK, font=font_caption)
        y_text += line_h

    # ── Figure caption ────────────────────────────────────────────────
    y_caption = y_answer + answer_h + PADDING // 2
    caption = f'Figure: Retrieved panels for query — "{query}"'
    draw.text((PADDING, y_caption), caption, fill=TEXT_MED, font=font_caption)

    # Outer border
    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=BORDER, width=2)

    query_slug = query[:30].replace(' ', '_').replace('?', '')
    model_slug = model.replace(':', '-')
    fname = f'{out_dir}/{model_slug}__{query_slug}.jpg'
    canvas.save(fname, quality=97)
    return fname


# def evaluate():
#     print('Retrieving panels for all queries...')
#     query_panels = {}
#     for query in TEST_QUERIES:
#         panels = retrieve_panels('old_boy_vol01', query, n_results=5)
#         query_panels[query] = panels
#         print(f'  "{query[:45]}" → {len(panels)} panels')

#     all_results = []
#     grid_path = save_panel_grid(
#                     panels, query, model
#                     )
#     tqdm.write(f'  Grid saved: {grid_path}')
#     total_runs = len(TEST_QUERIES) * len(MODELS)

#     with tqdm(total=total_runs, desc='Evaluating', unit='run') as pbar:
#         for query in TEST_QUERIES:
#             panels = query_panels[query]
#             for model in MODELS:
#                 pbar.set_description(f'{model[:20]} | {query[:25]}')
#                 try:
#                     r = run_model(model, query, panels)
#                     all_results.append(r)
#                     pbar.set_postfix({
#                         'latency': f'{r["total_latency_s"]}s',
#                         'tok/s': r['tokens_per_second'],
#                         'ref_rate': r['panel_reference_rate'],
#                     })
#                     tqdm.write(
#                         f'\n[{model}] "{query[:40]}"\n'
#                         f'  Latency:        {r["total_latency_s"]}s\n'
#                         f'  Prompt time:    {r["prompt_eval_time_s"]}s\n'
#                         f'  Gen time:       {r["generation_time_s"]}s\n'
#                         f'  Input tokens:   {r["input_tokens"]}\n'
#                         f'  Output tokens:  {r["output_tokens"]}\n'
#                         f'  Tokens/sec:     {r["tokens_per_second"]}\n'
#                         f'  I/O ratio:      {r["input_output_ratio"]}\n'
#                         f'  Panel ref rate: {r["panel_reference_rate"]}\n'
#                         f'  Answer: {r["answer"][:300]}...'
#                     )
#                 except Exception as e:
#                     tqdm.write(f'ERROR [{model}]: {e}')
#                 finally:
#                     pbar.update(1)


def evaluate():
    print('Retrieving panels for all queries...')
    query_panels = {}
    for query in TEST_QUERIES:
        panels = retrieve_panels('old_boy_vol01', query, n_results=5)
        query_panels[query] = panels
        print(f'  "{query[:45]}" → {len(panels)} panels')

    all_results = []
    total_runs = len(TEST_QUERIES) * len(MODELS)

    with tqdm(total=total_runs, desc='Evaluating', unit='run') as pbar:
        for query in TEST_QUERIES:
            panels = query_panels[query]
            for model in MODELS:
                pbar.set_description(f'{model[:20]} | {query[:25]}')
                try:
                    r = run_model(model, query, panels)
                    all_results.append(r)
                    pbar.set_postfix({
                        'latency': f'{r["total_latency_s"]}s',
                        'tok/s': r['tokens_per_second'],
                        'ref_rate': r['panel_reference_rate'],
                    })
                    grid_path = save_panel_grid(panels, query, model,r['answer'])
                    tqdm.write(
                        f'\n[{model}] "{query[:40]}"\n'
                        f'  Latency:        {r["total_latency_s"]}s\n'
                        f'  Prompt time:    {r["prompt_eval_time_s"]}s\n'
                        f'  Gen time:       {r["generation_time_s"]}s\n'
                        f'  Input tokens:   {r["input_tokens"]}\n'
                        f'  Output tokens:  {r["output_tokens"]}\n'
                        f'  Tokens/sec:     {r["tokens_per_second"]}\n'
                        f'  I/O ratio:      {r["input_output_ratio"]}\n'
                        f'  Panel ref rate: {r["panel_reference_rate"]}\n'
                        f'  Grid saved:     {grid_path}\n'
                        f'  Answer: {r["answer"][:300]}...'
                    )
                except Exception as e:
                    tqdm.write(f'ERROR [{model}]: {e}')
                finally:
                    pbar.update(1)

    print_divider()
    print('AVERAGES PER MODEL')
    print_divider()

    for model in MODELS:
        model_results = [r for r in all_results if r['model'] == model]
        if not model_results:
            print(f'\n{model} — no results (all errored)')
            continue

        def avg(key):
            return round(
                sum(r[key] for r in model_results) / len(model_results), 2
            )

        print(f'\n{model}')
        print(f'  Avg total latency:      {avg("total_latency_s")}s')
        print(f'  Avg prompt eval time:   {avg("prompt_eval_time_s")}s')
        print(f'  Avg generation time:    {avg("generation_time_s")}s')
        print(f'  Avg input tokens:       {avg("input_tokens")}')
        print(f'  Avg output tokens:      {avg("output_tokens")}')
        print(f'  Avg tokens/sec:         {avg("tokens_per_second")}')
        print(f'  Avg input/output ratio: {avg("input_output_ratio")}')
        print(f'  Avg panel ref rate:     {avg("panel_reference_rate")}')

    print_divider()





if __name__ == '__main__':
    evaluate()