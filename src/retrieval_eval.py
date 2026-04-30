import os
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from src.retriever import retrieve_panels, retrieve_panels_for_character

load_dotenv()

PANELS_DIR = Path(os.getenv('PANELS_DIR'))
OUTPUT_DIR = Path('data/retrieval_eval')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANGA_NAME = 'old_boy_vol01'
CHARACTERS = ['Shinichi Goto', 'Takaaki Kakinuma', 'Eri']

TEST_QUERIES = [
    'looking angry or intense',
    'two people fighting',
    'a person looking sad or defeated',
    'a dark threatening scene',
    'a person running or in danger',
]


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


def load_panel(panel_meta: dict, size: int = 200) -> Image.Image:
    try:
        img = Image.open(panel_meta['path']).convert('RGB')
        ratio = size / max(img.width, img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img   = img.resize((new_w, new_h))
        # Pad to square
        canvas = Image.new('RGB', (size, size), (230, 230, 230))
        x_off  = (size - new_w) // 2
        y_off  = (size - new_h) // 2
        canvas.paste(img, (x_off, y_off))
        return canvas
    except Exception:
        return Image.new('RGB', (size, size), (200, 200, 200))


def build_comparison_grid(
    query: str,
    standard_results: list[dict],
    character: str,
    character_results: list[dict],
) -> Image.Image:
    fonts    = get_fonts()
    PANEL_SZ = 180
    PADDING  = 12
    HEADER_H = 50
    LABEL_H  = 36
    N        = max(len(standard_results), len(character_results))

    BG       = (248, 248, 248)
    DARK     = (20,  20,  20)
    MED      = (90,  90,  90)
    ACCENT   = (30,  80,  160)
    GREEN    = (30,  130, 80)
    DIVIDER  = (180, 180, 180)

    section_w = PADDING + N * (PANEL_SZ + PADDING)
    total_w   = max(section_w * 2 + PADDING, 900)
    total_h   = HEADER_H + LABEL_H + PANEL_SZ + PADDING * 3 + 80

    canvas = Image.new('RGB', (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # Header — query
    draw.rectangle([0, 0, total_w, HEADER_H], fill=(25, 25, 25))
    draw.text((PADDING, 10), f'Query: "{query}"', fill=(255, 255, 255), font=fonts['title'])
    draw.text((PADDING, 32), f'Manga: {MANGA_NAME}  |  n_results: {len(standard_results)}',
              fill=(160, 160, 160), font=fonts['small'])

    half_w = total_w // 2

    # Section headers
    y_section = HEADER_H + PADDING

    draw.rectangle([0, y_section, half_w, y_section + LABEL_H], fill=ACCENT)
    draw.text((PADDING, y_section + 10), 'Standard Retrieval (all panels)',
              fill=(255, 255, 255), font=fonts['bold'])

    draw.rectangle([half_w, y_section, total_w, y_section + LABEL_H], fill=GREEN)
    draw.text((half_w + PADDING, y_section + 10),
              f'Character-Aware: {character}',
              fill=(255, 255, 255), font=fonts['bold'])

    # Vertical divider
    draw.line([(half_w, HEADER_H), (half_w, total_h)], fill=DIVIDER, width=2)

    # Panel rows
    y_panels = y_section + LABEL_H + PADDING

    for col, (results, x_base) in enumerate([
        (standard_results, 0),
        (character_results, half_w),
    ]):
        x = x_base + PADDING
        for i, r in enumerate(results):
            img = load_panel(r, size=PANEL_SZ)
            canvas.paste(img, (x, y_panels))

            # Similarity badge
            badge_text = f'{r["similarity"]:.3f}'
            draw.rectangle(
                [x, y_panels, x + 52, y_panels + 18],
                fill=(0, 0, 0, 180)
            )
            draw.text((x + 2, y_panels + 2), badge_text,
                      fill=(255, 220, 0), font=fonts['small'])

            # Page/panel label below
            label = f'p{r["page"]:04d} | pan{r["panel"]:02d}'
            draw.text((x, y_panels + PANEL_SZ + 2), label,
                      fill=MED, font=fonts['small'])

            x += PANEL_SZ + PADDING

    # Metrics bar at bottom
    y_metrics = y_panels + PANEL_SZ + 22

    std_sims  = [r['similarity'] for r in standard_results]
    char_sims = [r['similarity'] for r in character_results]

    std_avg   = np.mean(std_sims)  if std_sims  else 0
    char_avg  = np.mean(char_sims) if char_sims else 0
    std_max   = np.max(std_sims)   if std_sims  else 0
    char_max  = np.max(char_sims)  if char_sims else 0

    draw.rectangle([0, y_metrics, total_w, total_h], fill=(235, 235, 240))
    draw.text(
        (PADDING, y_metrics + 8),
        f'Standard  ->  avg sim: {std_avg:.3f}  |  max sim: {std_max:.3f}  |  panels searched: 1250',
        fill=DARK, font=fonts['regular']
    )
    draw.text(
        (PADDING, y_metrics + 28),
        f'Char-Aware ->  avg sim: {char_avg:.3f}  |  max sim: {char_max:.3f}  |  panels searched: registry only',
        fill=DARK, font=fonts['regular']
    )
    draw.text(
        (PADDING, y_metrics + 48),
        f'Similarity delta: {char_avg - std_avg:+.3f}  (negative = precision/recall tradeoff)',
        fill=MED, font=fonts['small']
    )

    # Outer border
    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=DIVIDER, width=2)

    return canvas


def run_evaluation():
    print('Running retrieval evaluation...')
    all_grids = []

    for character in CHARACTERS:
        print(f'\nCharacter: {character}')
        char_grids = []

        for query in TEST_QUERIES:
            print(f'  Query: "{query[:45]}"')

            std_results  = retrieve_panels(MANGA_NAME, query, n_results=5)
            char_results = retrieve_panels_for_character(
                MANGA_NAME, character, query, n_results=5
            )

            std_avg  = np.mean([r['similarity'] for r in std_results])
            char_avg = np.mean([r['similarity'] for r in char_results])

            print(f'    Standard avg sim:   {std_avg:.3f}')
            print(f'    Char-aware avg sim: {char_avg:.3f}')
            print(f'    Delta:              {char_avg - std_avg:+.3f}')

            grid = build_comparison_grid(query, std_results, character, char_results)
            char_grids.append((query, grid))

        # Stack all query grids for this character vertically
        total_h = sum(g.height for _, g in char_grids) + len(char_grids) * 8
        total_w = max(g.width for _, g in char_grids)
        stacked = Image.new('RGB', (total_w, total_h), (200, 200, 200))
        y = 0
        for _, g in char_grids:
            stacked.paste(g, (0, y))
            y += g.height + 8

        char_slug = character.replace(' ', '_')
        out_path  = OUTPUT_DIR / f'{char_slug}_retrieval_eval.jpg'
        stacked.save(out_path, quality=95)
        print(f'  Saved: {out_path}')
        all_grids.append(out_path)

    # Print summary table
    print(f'\n{"="*65}')
    print('RETRIEVAL EVALUATION SUMMARY')
    print('='*65)
    print(f'{"Character":<20} {"Query":<35} {"Std":>6} {"Char":>6} {"Delta":>7}')
    print('-'*65)

    for character in CHARACTERS:
        for query in TEST_QUERIES:
            std_results  = retrieve_panels(MANGA_NAME, query, n_results=5)
            char_results = retrieve_panels_for_character(
                MANGA_NAME, character, query, n_results=5
            )
            std_avg  = np.mean([r['similarity'] for r in std_results])
            char_avg = np.mean([r['similarity'] for r in char_results])
            print(f'{character:<20} {query[:33]:<35} '
                  f'{std_avg:>6.3f} {char_avg:>6.3f} {char_avg - std_avg:>+7.3f}')
        print()

    print('='*65)
    print(f'\nGrids saved to: {OUTPUT_DIR}')
    print('Open with: eog data/retrieval_eval/')


if __name__ == '__main__':
    run_evaluation()