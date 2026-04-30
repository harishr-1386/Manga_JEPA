import time
import base64
import os
import textwrap
import ollama
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from src.retriever import retrieve_panels

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

    input_tokens     = response.get('prompt_eval_count', 0)
    output_tokens    = response.get('eval_count', 0)
    eval_duration_s  = response.get('eval_duration', 0) / 1e9
    prompt_duration_s = response.get('prompt_eval_duration', 0) / 1e9
    tokens_per_second = round(output_tokens / eval_duration_s, 2) if eval_duration_s > 0 else 0

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


def get_fonts():
    try:
        return {
            'regular': ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13),
            'bold':    ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13),
            'title':   ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15),
            'caption': ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12),
            'small':   ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12),
        }
    except Exception:
        default = ImageFont.load_default()
        return {k: default for k in ['regular', 'bold', 'title', 'caption', 'small']}


def save_comparison_grid(
    panels: list[dict],
    query: str,
    results: dict,           # {model_name: result_dict}
    out_dir: str = 'data/eval_grids'
):
    """
    Single grid per query showing:
    - Panels once at the top
    - Both model responses + metrics side by side below
    """
    os.makedirs(out_dir, exist_ok=True)
    fonts = get_fonts()

    # --- Constants ---
    PANEL_H    = 360
    LABEL_H    = 22
    SUBLABEL_H = 36
    PADDING    = 16
    GAP        = 6
    LINE_H     = 17

    BG          = (255, 255, 255)
    BORDER      = (180, 180, 180)
    TEXT_DARK   = (20,  20,  20)
    TEXT_MED    = (90,  90,  90)
    ACCENT      = (30,  80,  160)
    TABLE_HEAD  = (235, 240, 250)
    MODEL_COLS  = [(245, 250, 255), (250, 255, 245)]  # blue tint / green tint per model

    # --- Resize panels ---
    images = [Image.open(p['path']).convert('RGB') for p in panels]
    resized = []
    for img in images:
        ratio = PANEL_H / img.height
        resized.append(img.resize((max(int(img.width * ratio), 80), PANEL_H)))

    n = len(resized)
    panel_row_w = sum(img.width for img in resized) + GAP * (n - 1)

    # --- Canvas width ---
    total_w = max(panel_row_w + PADDING * 2, 1100)
    half_w  = (total_w - PADDING * 3) // 2  # width of each model column

    # --- Answer wrapping per model ---
    chars_per_col = (half_w - PADDING * 2) // 7
    wrapped_answers = {}
    answer_heights  = {}
    for model, r in results.items():
        wrapped = textwrap.fill(r['answer'], width=chars_per_col)
        lines   = wrapped.split('\n')
        wrapped_answers[model] = lines
        answer_heights[model]  = len(lines) * LINE_H

    # Metrics table: 4 rows x 22px + padding
    table_h   = 22 * 5 + PADDING
    max_ans_h = max(answer_heights.values()) + PADDING * 3 + 20  # +header

    model_section_h = table_h + max_ans_h + PADDING

    # --- Total canvas height ---
    panel_section_h = PADDING + LABEL_H + PANEL_H + SUBLABEL_H + PADDING
    divider_h       = 32   # query banner
    caption_h       = 36
    total_h = panel_section_h + divider_h + model_section_h + caption_h + PADDING

    canvas = Image.new('RGB', (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # --- Panel row ---
    x        = PADDING
    y_label  = PADDING
    y_img    = y_label + LABEL_H
    y_sub    = y_img + PANEL_H

    for i, (img, panel) in enumerate(zip(resized, panels)):
        letter = f'({chr(97 + i)})'
        draw.text(
            (x + img.width // 2 - 8, y_label),
            letter, fill=ACCENT, font=fonts['bold']
        )
        canvas.paste(img, (x, y_img))
        draw.rectangle([x - 1, y_img - 1, x + img.width, y_img + PANEL_H],
                       outline=BORDER, width=1)

        draw.text((x, y_sub + 2),  f'Page {panel["page"]:04d}, Panel {panel["panel"]:02d}',
                  fill=TEXT_MED,  font=fonts['small'])
        draw.text((x, y_sub + 17), f'Similarity: {panel["similarity"]:.3f}',
                  fill=TEXT_DARK, font=fonts['bold'])
        x += img.width + GAP

    # --- Query banner ---
    y_banner = panel_section_h
    draw.rectangle([0, y_banner, total_w, y_banner + divider_h], fill=(30, 30, 30))
    draw.text(
        (PADDING, y_banner + 8),
        f'Query: "{query}"',
        fill=(255, 255, 255), font=fonts['title']
    )

    # --- Model columns ---
    y_model_top = y_banner + divider_h + PADDING

    for col_i, (model, r) in enumerate(results.items()):
        x_col = PADDING + col_i * (half_w + PADDING)
        bg_col = MODEL_COLS[col_i % 2]

        # Column background
        draw.rectangle(
            [x_col, y_model_top, x_col + half_w, y_model_top + model_section_h - PADDING],
            fill=bg_col, outline=BORDER, width=1
        )

        # Model name header
        draw.rectangle(
            [x_col, y_model_top, x_col + half_w, y_model_top + 26],
            fill=ACCENT
        )
        draw.text(
            (x_col + 8, y_model_top + 6),
            model, fill=(255, 255, 255), font=fonts['title']
        )

        # Metrics table
        y_table = y_model_top + 30
        metrics = [
            ('Latency',        f'{r["total_latency_s"]}s'),
            ('Input tokens',   str(r['input_tokens'])),
            ('Output tokens',  str(r['output_tokens'])),
            ('Tokens/sec',     str(r['tokens_per_second'])),
            ('Panel ref rate', str(r['panel_reference_rate'])),
        ]

        # Table header
        draw.rectangle(
            [x_col + 4, y_table, x_col + half_w - 4, y_table + 20],
            fill=TABLE_HEAD, outline=BORDER, width=1
        )
        draw.text((x_col + 8,            y_table + 3), 'Metric',  fill=TEXT_DARK, font=fonts['bold'])
        draw.text((x_col + half_w // 2,  y_table + 3), 'Value',   fill=TEXT_DARK, font=fonts['bold'])

        for row_i, (metric, value) in enumerate(metrics):
            row_y  = y_table + 20 + row_i * 22
            row_bg = (255, 255, 255) if row_i % 2 == 0 else bg_col
            draw.rectangle(
                [x_col + 4, row_y, x_col + half_w - 4, row_y + 22],
                fill=row_bg, outline=BORDER, width=1
            )
            draw.text((x_col + 8,           row_y + 4), metric, fill=TEXT_DARK, font=fonts['regular'])
            draw.text((x_col + half_w // 2, row_y + 4), value,  fill=TEXT_DARK, font=fonts['bold'])

        # Answer box
        y_ans_box = y_table + 20 + len(metrics) * 22 + PADDING // 2
        draw.text(
            (x_col + 8, y_ans_box),
            'Model Response:',
            fill=ACCENT, font=fonts['title']
        )
        y_text = y_ans_box + 20
        for line in wrapped_answers[model]:
            draw.text((x_col + 8, y_text), line, fill=TEXT_DARK, font=fonts['caption'])
            y_text += LINE_H

    # --- Figure caption ---
    y_caption = y_model_top + model_section_h
    avg_sim   = round(sum(p['similarity'] for p in panels) / len(panels), 3)
    caption   = (f'Figure: Model comparison for query: "{query}" | '
                 f'Panels: {len(panels)} | Avg similarity: {avg_sim}')
    draw.text((PADDING, y_caption + 8), caption, fill=TEXT_MED, font=fonts['caption'])

    # Outer border
    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=BORDER, width=2)

    query_slug = query[:35].replace(' ', '_').replace('?', '')
    fname = f'{out_dir}/comparison__{query_slug}.jpg'
    canvas.save(fname, quality=97)
    return fname




def print_divider(char='=', width=70):
    print(char * width)


def evaluate():
    print('Retrieving panels for all queries...')
    query_panels = {}
    for query in TEST_QUERIES:
        panels = retrieve_panels('old_boy_vol01', query, n_results=5)
        query_panels[query] = panels
        print(f'  "{query[:45]}" -> {len(panels)} panels')

    all_results = []
    total_runs  = len(TEST_QUERIES) * len(MODELS)

    with tqdm(total=total_runs, desc='Evaluating', unit='run') as pbar:
        for query in TEST_QUERIES:
            panels          = query_panels[query]
            query_results   = {}   # collect both models before saving grid

            for model in MODELS:
                pbar.set_description(f'{model[:20]} | {query[:25]}')
                try:
                    r = run_model(model, query, panels)
                    all_results.append(r)
                    query_results[model] = r
                    pbar.set_postfix({
                        'latency': f'{r["total_latency_s"]}s',
                        'tok/s':   r['tokens_per_second'],
                        'ref':     r['panel_reference_rate'],
                    })
                    tqdm.write(
                        f'\n[{model}] "{query[:40]}"\n'
                        f'  Latency:        {r["total_latency_s"]}s\n'
                        f'  Input tokens:   {r["input_tokens"]}\n'
                        f'  Output tokens:  {r["output_tokens"]}\n'
                        f'  Tokens/sec:     {r["tokens_per_second"]}\n'
                        f'  Panel ref rate: {r["panel_reference_rate"]}\n'
                        f'  Answer: {r["answer"][:200]}...'
                    )
                except Exception as e:
                    tqdm.write(f'ERROR [{model}]: {e}')
                finally:
                    pbar.update(1)

            # Save one comparison grid per query once both models are done
            if len(query_results) == len(MODELS):
                grid_path = save_comparison_grid(panels, query, query_results)
                tqdm.write(f'\n  Comparison grid saved: {grid_path}')

    print_divider()
    print('AVERAGES PER MODEL')
    print_divider()

    for model in MODELS:
        model_results = [r for r in all_results if r['model'] == model]
        if not model_results:
            print(f'\n{model} - no results (all errored)')
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