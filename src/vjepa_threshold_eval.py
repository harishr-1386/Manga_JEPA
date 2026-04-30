import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_DIR = Path(os.getenv('EMBEDDINGS_DIR'))
MANGA_NAME     = 'old_boy_vol01'
REGISTRY_PATH  = Path(f'data/labels/{MANGA_NAME}_character_registry.json')
OUTPUT_DIR     = Path('data/vjepa_threshold_eval')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHARACTERS   = ['Shinichi Goto', 'Takaaki Kakinuma', 'Eri']
THRESHOLDS   = [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
MAX_NEIGHBORS = 15
TOTAL_PANELS  = 1250


def load_data():
    clip_emb  = np.load(EMBEDDINGS_DIR / MANGA_NAME / 'clip_embeddings.npy')
    vjepa_emb = np.load(EMBEDDINGS_DIR / MANGA_NAME / 'embeddings.npy')
    with open(EMBEDDINGS_DIR / MANGA_NAME / 'metadata.json') as f:
        metadata = json.load(f)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    panel_ids       = [m['panel_id'] for m in metadata]
    panel_id_to_idx = {pid: i for i, pid in enumerate(panel_ids)}
    return clip_emb, vjepa_emb, panel_ids, panel_id_to_idx, registry


def normalize(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms


def propagate(emb_norm, panel_ids, panel_id_to_idx, registry, threshold, max_neighbors):
    propagated  = dict(registry)
    vote_counts = defaultdict(lambda: defaultdict(float))

    for panel_id, character in registry.items():
        if panel_id not in panel_id_to_idx:
            continue
        idx   = panel_id_to_idx[panel_id]
        sims  = emb_norm @ emb_norm[idx]
        top   = np.argsort(sims)[::-1]
        added = 0
        for n_idx in top:
            if n_idx == idx:
                continue
            if sims[n_idx] < threshold:
                break
            if added >= max_neighbors:
                break
            vote_counts[panel_ids[n_idx]][character] += float(sims[n_idx])
            added += 1

    newly = 0
    for pid, votes in vote_counts.items():
        if pid not in propagated:
            propagated[pid] = max(votes, key=votes.get)
            newly += 1
    return propagated, newly


def loo_error(emb_norm, panel_id_to_idx, registry, threshold):
    correct, wrong = 0, 0
    for held_id, true_label in registry.items():
        if held_id not in panel_id_to_idx:
            continue
        held_idx    = panel_id_to_idx[held_id]
        reduced     = {k: v for k, v in registry.items() if k != held_id}
        vote_counts = defaultdict(float)
        for pid, char in reduced.items():
            if pid not in panel_id_to_idx:
                continue
            idx = panel_id_to_idx[pid]
            sim = float(emb_norm[held_idx] @ emb_norm[idx])
            if sim >= threshold:
                vote_counts[char] += sim
        if vote_counts:
            predicted = max(vote_counts, key=vote_counts.get)
            if predicted == true_label:
                correct += 1
            else:
                wrong += 1
    total = correct + wrong
    acc = correct / total if total > 0 else 0
    err = wrong   / total if total > 0 else 0
    return round(acc, 4), round(err, 4)


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


def build_comparison_viz(clip_results, vjepa_results):
    fonts    = get_fonts()
    PADDING  = 16
    HEADER_H = 70
    ROW_H    = 100
    LEFT_W   = 180
    BAR_W    = 380
    METRIC_W = 200
    total_w  = LEFT_W + BAR_W * 2 + METRIC_W + PADDING * 4
    total_h  = HEADER_H + len(THRESHOLDS) * ROW_H + PADDING * 2 + 50

    BG      = (248, 248, 248)
    DARK    = (20,  20,  20)
    MED     = (90,  90,  90)
    CLIP_C  = (40,  80,  180)
    VJEPA_C = (180, 60,  40)
    GREEN   = (40,  160, 80)
    ERROR_C = (200, 100, 30)

    CHAR_COLORS = {
        'Shinichi Goto':    (50,  100, 200),
        'Takaaki Kakinuma': (200, 60,  60),
        'Eri':              (60,  160, 80),
    }

    canvas = Image.new('RGB', (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([0, 0, total_w, HEADER_H], fill=(20, 20, 20))
    draw.text((PADDING, 10), 'Label Propagation Threshold: CLIP vs V-JEPA 2',
              fill=(255, 255, 255), font=fonts['title'])
    draw.text((PADDING, 34), f'Manga: {MANGA_NAME}  |  Total panels: {TOTAL_PANELS}  |  Max neighbors: {MAX_NEIGHBORS}',
              fill=(160, 160, 160), font=fonts['small'])

    # Column headers
    y_col = HEADER_H + 4
    draw.text((PADDING, y_col), 'Threshold', fill=DARK, font=fonts['bold'])

    clip_x = LEFT_W + PADDING
    draw.rectangle([clip_x, y_col - 2, clip_x + BAR_W, y_col + 18], fill=CLIP_C)
    draw.text((clip_x + 8, y_col), 'CLIP Coverage + Error Rate',
              fill=(255, 255, 255), font=fonts['bold'])

    vjepa_x = clip_x + BAR_W + PADDING
    draw.rectangle([vjepa_x, y_col - 2, vjepa_x + BAR_W, y_col + 18], fill=VJEPA_C)
    draw.text((vjepa_x + 8, y_col), 'V-JEPA 2 Coverage + Error Rate',
              fill=(255, 255, 255), font=fonts['bold'])

    max_total = max(
        max(r['clip_total'] for r in clip_results),
        max(r['vjepa_total'] for r in vjepa_results)
    )

    for i, (cr, vr) in enumerate(zip(clip_results, vjepa_results)):
        y      = HEADER_H + 28 + i * ROW_H
        thresh = cr['threshold']

        # Highlight sweet spots
        clip_sweet  = cr.get('sweet_spot', False)
        vjepa_sweet = vr.get('sweet_spot', False)
        if clip_sweet or vjepa_sweet:
            draw.rectangle([0, y, total_w, y + ROW_H - 4], fill=(240, 248, 240))

        # Threshold label
        draw.text((PADDING, y + 8),  f'{thresh:.2f}', fill=DARK, font=fonts['bold'])
        draw.text((PADDING, y + 26), f'CLIP: {cr["clip_total"]}', fill=CLIP_C,  font=fonts['small'])
        draw.text((PADDING, y + 40), f'VJEPA: {vr["vjepa_total"]}', fill=VJEPA_C, font=fonts['small'])

        if clip_sweet:
            draw.text((PADDING, y + 56), 'CLIP SWEET', fill=GREEN, font=fonts['small'])
        if vjepa_sweet:
            draw.text((PADDING, y + 68), 'VJEPA SWEET', fill=GREEN, font=fonts['small'])

        # CLIP bars
        x = clip_x
        bar_y = y + 8
        bar_h = 16
        for char in CHARACTERS:
            count = cr['clip_counts'].get(char, 0)
            color = CHAR_COLORS[char]
            w     = int(BAR_W * 0.7 * (count / max_total)) if max_total > 0 else 0
            if w > 0:
                draw.rectangle([x, bar_y, x + w, bar_y + bar_h], fill=color)
                if w > 25:
                    draw.text((x + 3, bar_y + 2), str(count),
                              fill=(255, 255, 255), font=fonts['small'])
            bar_y += bar_h + 3

        # CLIP error bar
        err_w = int(BAR_W * 0.7 * cr['clip_err'])
        draw.rectangle([x, bar_y, x + int(BAR_W * 0.7), bar_y + bar_h], fill=(220, 220, 220))
        if err_w > 0:
            draw.rectangle([x, bar_y, x + err_w, bar_y + bar_h], fill=ERROR_C)
        draw.text((x + int(BAR_W * 0.7) + 4, bar_y + 2),
                  f'err:{cr["clip_err"]*100:.0f}%  acc:{cr["clip_acc"]*100:.0f}%',
                  fill=ERROR_C, font=fonts['small'])

        # V-JEPA bars
        x     = vjepa_x
        bar_y = y + 8
        for char in CHARACTERS:
            count = vr['vjepa_counts'].get(char, 0)
            color = CHAR_COLORS[char]
            w     = int(BAR_W * 0.7 * (count / max_total)) if max_total > 0 else 0
            if w > 0:
                draw.rectangle([x, bar_y, x + w, bar_y + bar_h], fill=color)
                if w > 25:
                    draw.text((x + 3, bar_y + 2), str(count),
                              fill=(255, 255, 255), font=fonts['small'])
            bar_y += bar_h + 3

        # V-JEPA error bar
        err_w = int(BAR_W * 0.7 * vr['vjepa_err'])
        draw.rectangle([x, bar_y, x + int(BAR_W * 0.7), bar_y + bar_h], fill=(220, 220, 220))
        if err_w > 0:
            draw.rectangle([x, bar_y, x + err_w, bar_y + bar_h], fill=ERROR_C)
        draw.text((x + int(BAR_W * 0.7) + 4, bar_y + 2),
                  f'err:{vr["vjepa_err"]*100:.0f}%  acc:{vr["vjepa_acc"]*100:.0f}%',
                  fill=ERROR_C, font=fonts['small'])

        # Row divider
        draw.line([(0, y + ROW_H - 4), (total_w, y + ROW_H - 4)],
                  fill=(210, 210, 210), width=1)

    # Legend
    y_leg = HEADER_H + 28 + len(THRESHOLDS) * ROW_H + 8
    draw.rectangle([0, y_leg, total_w, total_h], fill=(235, 235, 240))
    x_leg = PADDING
    for char in CHARACTERS:
        draw.rectangle([x_leg, y_leg + 18, x_leg + 16, y_leg + 32], fill=CHAR_COLORS[char])
        draw.text((x_leg + 20, y_leg + 18), char, fill=DARK, font=fonts['small'])
        x_leg += len(char) * 7 + 30
    draw.rectangle([x_leg, y_leg + 18, x_leg + 16, y_leg + 32], fill=ERROR_C)
    draw.text((x_leg + 20, y_leg + 18), 'Error Rate', fill=DARK, font=fonts['small'])

    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(180, 180, 180), width=2)
    return canvas


def run():
    print('Loading data...')
    clip_emb, vjepa_emb, panel_ids, panel_id_to_idx, registry = load_data()
    clip_norm  = normalize(clip_emb)
    vjepa_norm = normalize(vjepa_emb)

    print(f'\n{"Thresh":>8} | {"CLIP Total":>10} {"CLIP Acc":>9} {"CLIP Err":>9} | {"VJEPA Total":>11} {"VJEPA Acc":>10} {"VJEPA Err":>10}')
    print('-' * 80)

    clip_results  = []
    vjepa_results = []

    for threshold in THRESHOLDS:
        clip_prop,  _ = propagate(clip_norm,  panel_ids, panel_id_to_idx, registry, threshold, MAX_NEIGHBORS)
        vjepa_prop, _ = propagate(vjepa_norm, panel_ids, panel_id_to_idx, registry, threshold, MAX_NEIGHBORS)

        clip_acc,  clip_err  = loo_error(clip_norm,  panel_id_to_idx, registry, threshold)
        vjepa_acc, vjepa_err = loo_error(vjepa_norm, panel_id_to_idx, registry, threshold)

        clip_counts  = Counter(clip_prop.values())
        vjepa_counts = Counter(vjepa_prop.values())
        clip_total   = sum(clip_counts.get(c, 0)  for c in CHARACTERS)
        vjepa_total  = sum(vjepa_counts.get(c, 0) for c in CHARACTERS)

        print(f'{threshold:>8.2f} | {clip_total:>10} {clip_acc*100:>8.1f}% {clip_err*100:>8.1f}% | '
              f'{vjepa_total:>11} {vjepa_acc*100:>9.1f}% {vjepa_err*100:>9.1f}%')

        clip_results.append({
            'threshold': threshold, 'clip_total': clip_total,
            'clip_acc': clip_acc, 'clip_err': clip_err,
            'clip_counts': dict(clip_counts),
        })
        vjepa_results.append({
            'threshold': threshold, 'vjepa_total': vjepa_total,
            'vjepa_acc': vjepa_acc, 'vjepa_err': vjepa_err,
            'vjepa_counts': dict(vjepa_counts),
        })

    # Find sweet spots (max coverage under 20% error)
    clip_valid  = [r for r in clip_results  if r['clip_err']  < 0.20]
    vjepa_valid = [r for r in vjepa_results if r['vjepa_err'] < 0.20]

    if clip_valid:
        best_clip = max(clip_valid, key=lambda r: r['clip_total'])
        best_clip['sweet_spot'] = True
        print(f'\nCLIP sweet spot:    threshold={best_clip["threshold"]:.2f} '
              f'total={best_clip["clip_total"]} err={best_clip["clip_err"]*100:.1f}%')

    if vjepa_valid:
        best_vjepa = max(vjepa_valid, key=lambda r: r['vjepa_total'])
        best_vjepa['sweet_spot'] = True
        print(f'V-JEPA 2 sweet spot: threshold={best_vjepa["threshold"]:.2f} '
              f'total={best_vjepa["vjepa_total"]} err={best_vjepa["vjepa_err"]*100:.1f}%')

    # Build visualization
    canvas   = build_comparison_viz(clip_results, vjepa_results)
    out_path = OUTPUT_DIR / 'clip_vs_vjepa_threshold.jpg'
    canvas.save(out_path, quality=95)
    print(f'\nVisualization saved: {out_path}')
    print('Open with: eog data/vjepa_threshold_eval/clip_vs_vjepa_threshold.jpg')


if __name__ == '__main__':
    run()