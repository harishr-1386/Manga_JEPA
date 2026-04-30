import os
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_DIR  = Path(os.getenv('EMBEDDINGS_DIR'))
MANGA_NAME      = 'old_boy_vol01'
REGISTRY_PATH   = Path(f'data/labels/{MANGA_NAME}_character_registry.json')
OUTPUT_DIR      = Path('data/threshold_eval')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHARACTERS      = ['Shinichi Goto', 'Takaaki Kakinuma', 'Eri']
THRESHOLDS      = [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
MAX_NEIGHBORS   = 15
TOTAL_PANELS    = 1250


def propagate_at_threshold(threshold: float) -> dict:
    clip_embeddings = np.load(EMBEDDINGS_DIR / MANGA_NAME / 'clip_embeddings.npy')
    with open(EMBEDDINGS_DIR / MANGA_NAME / 'metadata.json') as f:
        metadata = json.load(f)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    panel_ids       = [m['panel_id'] for m in metadata]
    panel_id_to_idx = {pid: i for i, pid in enumerate(panel_ids)}
    norms           = np.linalg.norm(clip_embeddings, axis=1, keepdims=True) + 1e-8
    clip_norm       = clip_embeddings / norms

    propagated  = dict(registry)
    vote_counts = defaultdict(lambda: defaultdict(float))

    for panel_id, character in registry.items():
        if panel_id not in panel_id_to_idx:
            continue
        idx       = panel_id_to_idx[panel_id]
        query_emb = clip_norm[idx]
        sims      = clip_norm @ query_emb
        top_indices = np.argsort(sims)[::-1]
        added = 0
        for neighbor_idx in top_indices:
            if neighbor_idx == idx:
                continue
            if sims[neighbor_idx] < threshold:
                break
            if added >= MAX_NEIGHBORS:
                break
            neighbor_id = panel_ids[neighbor_idx]
            vote_counts[neighbor_id][character] += float(sims[neighbor_idx])
            added += 1

    newly = 0
    for panel_id, votes in vote_counts.items():
        if panel_id not in propagated:
            best_char = max(votes, key=votes.get)
            propagated[panel_id] = best_char
            newly += 1

    return propagated, newly


def estimate_errors(propagated: dict, threshold: float) -> dict:
    """
    Estimate labeling errors using cross-validation on manual labels.
    For each manually labeled panel, temporarily remove it and re-propagate.
    Check if the propagated label matches the manual label.
    """
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    clip_embeddings = np.load(EMBEDDINGS_DIR / MANGA_NAME / 'clip_embeddings.npy')
    with open(EMBEDDINGS_DIR / MANGA_NAME / 'metadata.json') as f:
        metadata = json.load(f)

    panel_ids       = [m['panel_id'] for m in metadata]
    panel_id_to_idx = {pid: i for i, pid in enumerate(panel_ids)}
    norms           = np.linalg.norm(clip_embeddings, axis=1, keepdims=True) + 1e-8
    clip_norm       = clip_embeddings / norms

    correct = 0
    wrong   = 0
    tested  = 0

    for held_out_id, true_label in registry.items():
        if held_out_id not in panel_id_to_idx:
            continue

        # Build reduced registry without held-out panel
        reduced = {k: v for k, v in registry.items() if k != held_out_id}
        vote_counts = defaultdict(float)

        for panel_id, character in reduced.items():
            if panel_id not in panel_id_to_idx:
                continue
            idx       = panel_id_to_idx[panel_id]
            query_emb = clip_norm[idx]
            held_idx  = panel_id_to_idx[held_out_id]
            sim       = float(clip_norm[held_idx] @ query_emb)

            if sim >= threshold:
                vote_counts[character] += sim

        if vote_counts:
            predicted = max(vote_counts, key=vote_counts.get)
            if predicted == true_label:
                correct += 1
            else:
                wrong += 1
            tested += 1

    accuracy = correct / tested if tested > 0 else 0
    error_rate = wrong / tested if tested > 0 else 0

    return {
        'tested':     tested,
        'correct':    correct,
        'wrong':      wrong,
        'accuracy':   round(accuracy, 4),
        'error_rate': round(error_rate, 4),
    }


def get_fonts():
    try:
        return {
            'regular': ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13),
            'bold':    ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14),
            'title':   ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16),
            'small':   ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11),
        }
    except Exception:
        default = ImageFont.load_default()
        return {k: default for k in ['regular', 'bold', 'title', 'small']}


def draw_bar(draw, x, y, width, height, value, max_value, color, bg=(220,220,220)):
    draw.rectangle([x, y, x + width, y + height], fill=bg)
    filled = int(width * (value / max_value)) if max_value > 0 else 0
    if filled > 0:
        draw.rectangle([x, y, x + filled, y + height], fill=color)


def build_visualization(results: list[dict]) -> Image.Image:
    fonts   = get_fonts()

    CHAR_COLORS = {
        'Shinichi Goto':    (50,  100, 200),
        'Takaaki Kakinuma': (200, 60,  60),
        'Eri':              (60,  160, 80),
        'Total':            (80,  80,  80),
        'Error Rate':       (200, 120, 30),
    }

    PADDING  = 20
    ROW_H    = 90
    HEADER_H = 80
    LEGEND_H = 60
    BAR_W    = 500
    LEFT_COL = 200
    total_w  = LEFT_COL + BAR_W + PADDING * 3 + 160
    total_h  = HEADER_H + len(results) * ROW_H + LEGEND_H + PADDING * 2

    canvas = Image.new('RGB', (total_w, total_h), (248, 248, 248))
    draw   = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([0, 0, total_w, HEADER_H], fill=(25, 25, 25))
    draw.text((PADDING, 14), 'Label Propagation Threshold Evaluation', fill=(255,255,255), font=fonts['title'])
    draw.text((PADDING, 40), f'Manga: {MANGA_NAME}  |  Total panels: {TOTAL_PANELS}  |  Max neighbors: {MAX_NEIGHBORS}',
              fill=(160,160,160), font=fonts['small'])
    draw.text((PADDING, 56), 'Bars show panel count per character. Error rate from leave-one-out cross-validation.',
              fill=(120,120,120), font=fonts['small'])

    max_total = max(r['total'] for r in results)

    for row_i, r in enumerate(results):
        y_base = HEADER_H + PADDING + row_i * ROW_H
        is_sweet = r.get('sweet_spot', False)

        # Row background
        bg_color = (235, 245, 235) if is_sweet else (248, 248, 248)
        draw.rectangle([0, y_base, total_w, y_base + ROW_H - 4], fill=bg_color)

        if is_sweet:
            draw.rectangle([0, y_base, 4, y_base + ROW_H - 4], fill=(50, 180, 80))

        # Threshold label
        draw.text(
            (PADDING, y_base + 8),
            f'threshold={r["threshold"]:.2f}',
            fill=(20, 20, 20), font=fonts['bold']
        )
        draw.text(
            (PADDING, y_base + 28),
            f'total: {r["total"]}',
            fill=(80, 80, 80), font=fonts['small']
        )
        draw.text(
            (PADDING, y_base + 42),
            f'err: {r["error_rate"]*100:.1f}%',
            fill=CHAR_COLORS['Error Rate'], font=fonts['small']
        )
        draw.text(
            (PADDING, y_base + 56),
            f'acc: {r["accuracy"]*100:.1f}%',
            fill=(50, 150, 50), font=fonts['small']
        )

        if is_sweet:
            draw.text(
                (PADDING, y_base + 70),
                'Optimal Threshold',
                fill=(50, 150, 50), font=fonts['small']
            )

        # Stacked bars per character
        bar_y    = y_base + 8
        bar_h    = 18
        x_offset = LEFT_COL

        for char in CHARACTERS:
            count = r['counts'].get(char, 0)
            color = CHAR_COLORS[char]
            w     = int(BAR_W * (count / max_total)) if max_total > 0 else 0
            if w > 0:
                draw.rectangle([x_offset, bar_y, x_offset + w, bar_y + bar_h], fill=color)
                if w > 30:
                    draw.text(
                        (x_offset + 4, bar_y + 2),
                        str(count),
                        fill=(255, 255, 255), font=fonts['small']
                    )
            bar_y += bar_h + 4

        # Error rate bar
        err_w = int(BAR_W * r['error_rate'])
        draw.rectangle(
            [LEFT_COL, bar_y, LEFT_COL + BAR_W, bar_y + bar_h],
            fill=(220, 220, 220)
        )
        if err_w > 0:
            draw.rectangle(
                [LEFT_COL, bar_y, LEFT_COL + err_w, bar_y + bar_h],
                fill=CHAR_COLORS['Error Rate']
            )
        draw.text(
            (LEFT_COL + BAR_W + 8, bar_y + 2),
            f'{r["error_rate"]*100:.1f}% error',
            fill=CHAR_COLORS['Error Rate'], font=fonts['small']
        )

        # Divider
        draw.line(
            [(0, y_base + ROW_H - 4), (total_w, y_base + ROW_H - 4)],
            fill=(210, 210, 210), width=1
        )

    # Legend
    y_legend = HEADER_H + PADDING + len(results) * ROW_H + 10
    draw.rectangle([0, y_legend, total_w, total_h], fill=(235, 235, 240))

    x_leg = PADDING
    for char in CHARACTERS:
        color = CHAR_COLORS[char]
        draw.rectangle([x_leg, y_legend + 20, x_leg + 20, y_legend + 36], fill=color)
        draw.text((x_leg + 24, y_legend + 20), char, fill=(30,30,30), font=fonts['small'])
        x_leg += len(char) * 8 + 40

    draw.rectangle([x_leg, y_legend + 20, x_leg + 20, y_legend + 36], fill=CHAR_COLORS['Error Rate'])
    draw.text((x_leg + 24, y_legend + 20), 'Error Rate', fill=(30,30,30), font=fonts['small'])

    draw.rectangle([0, 0, total_w-1, total_h-1], outline=(180,180,180), width=2)

    return canvas


def run():
    print(f'Evaluating {len(THRESHOLDS)} thresholds...\n')
    print(f'{"Thresh":>8} {"Total":>7} {"Goto":>7} {"Kakinuma":>10} {"Eri":>6} {"Accuracy":>10} {"Error%":>8}')
    print('-' * 65)

    results = []

    for threshold in THRESHOLDS:
        propagated, newly = propagate_at_threshold(threshold)
        counts    = Counter(propagated.values())
        errors    = estimate_errors(propagated, threshold)
        total     = sum(counts.get(c, 0) for c in CHARACTERS)
        coverage  = total / TOTAL_PANELS

        result = {
            'threshold':  threshold,
            'total':      total,
            'coverage':   coverage,
            'counts':     dict(counts),
            'newly':      newly,
            'accuracy':   errors['accuracy'],
            'error_rate': errors['error_rate'],
            'tested':     errors['tested'],
        }
        results.append(result)

        print(
            f'{threshold:>8.2f} {total:>7} '
            f'{counts.get("Shinichi Goto", 0):>7} '
            f'{counts.get("Takaaki Kakinuma", 0):>10} '
            f'{counts.get("Eri", 0):>6} '
            f'{errors["accuracy"]*100:>9.1f}% '
            f'{errors["error_rate"]*100:>7.1f}%'
        )

    # Find sweet spot: maximize coverage while keeping error < 20%
    valid = [r for r in results if r['error_rate'] < 0.20]
    if valid:
        sweet = max(valid, key=lambda r: r['total'])
        sweet['sweet_spot'] = True
        print(f'\nSweet spot: threshold={sweet["threshold"]:.2f} '
              f'(total={sweet["total"]}, error={sweet["error_rate"]*100:.1f}%)')

        # Save the sweet spot propagated registry
        best_propagated, _ = propagate_at_threshold(sweet['threshold'])
        best_path = Path(f'data/labels/{MANGA_NAME}_propagated_registry.json')
        with open(best_path, 'w') as f:
            json.dump(best_propagated, f, indent=2)
        print(f'Sweet spot registry saved to {best_path}')
    else:
        print('\nNo threshold achieved <20% error rate.')

    # Build and save visualization
    canvas   = build_visualization(results)
    out_path = OUTPUT_DIR / 'threshold_evaluation.jpg'
    canvas.save(out_path, quality=95)
    print(f'\nVisualization saved to {out_path}')
    print('Open with: eog data/threshold_eval/threshold_evaluation.jpg')

    return results


if __name__ == '__main__':
    run()
