import zipfile
import io
import os
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO
import shutil
from dotenv import load_dotenv

load_dotenv()

PANEL_DETECTOR_PATH = os.getenv('PANEL_DETECTOR')
OUTPUT_DIR = Path(os.getenv('PANELS_DIR'))




def load_detector():
    model = YOLO(PANEL_DETECTOR_PATH)
    print('Panel detector loaded')
    return model


def extract_pages_from_cbz(cbz_path: str) -> list[tuple[str, Image.Image]]:
    """Extract all images from a CBZ file, sorted by filename."""
    pages = []
    with zipfile.ZipFile(cbz_path, 'r') as zf:
        names = sorted([
            n for n in zf.namelist()
            if n.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        for name in names:
            with zf.open(name) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
                pages.append((name, img))
    print(f'Extracted {len(pages)} pages from {Path(cbz_path).name}')
    return pages


def detect_and_crop_panels(
    detector: YOLO,
    pages: list[tuple[str, Image.Image]],
    manga_name: str,
    conf_threshold: float = 0.3,
) -> list[dict]:
    """Run YOLO on each page, crop detected panels, save to disk."""
    manga_out = OUTPUT_DIR / manga_name
    manga_out.mkdir(parents=True, exist_ok=True)

    all_panels = []
    for page_idx, (page_name, img) in enumerate(pages):
        results = detector(img, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            # No panels detected — treat whole page as one panel
            panel_path = manga_out / f'page{page_idx:04d}_panel0000.jpg'
            img.save(panel_path)
            all_panels.append({
                'manga': manga_name,
                'page': page_idx,
                'panel': 0,
                'path': str(panel_path),
                'bbox': [0, 0, img.width, img.height],
            })
            continue

        # Sort boxes top-to-bottom, left-to-right (reading order)
        xyxy = boxes.xyxy.cpu().numpy()
        sorted_indices = sorted(
            range(len(xyxy)),
            key=lambda i: (xyxy[i][1] // 100, xyxy[i][0])
        )

        # for panel_idx, i in enumerate(sorted_indices):
        #     x1, y1, x2, y2 = map(int, xyxy[i])
        #     crop = img.crop((x1, y1, x2, y2))
        #     panel_path = manga_out / f'page{page_idx:04d}_panel{panel_idx:04d}.jpg'
        #     crop.save(panel_path)
        #     all_panels.append({
        #         'manga': manga_name,
        #         'page': page_idx,
        #         'panel': panel_idx,
        #         'path': str(panel_path),
        #         'bbox': [x1, y1, x2, y2],
        #     })
        for panel_idx, i in enumerate(sorted_indices):
            x1, y1, x2, y2 = map(int, xyxy[i])
            # Skip tiny detections — noise/text artifacts
            if (x2 - x1) < 100 or (y2 - y1) < 100:
                continue
            crop = img.crop((x1, y1, x2, y2))
            panel_path = manga_out / f'page{page_idx:04d}_panel{panel_idx:04d}.jpg'
            crop.save(panel_path)
            all_panels.append({
                'manga': manga_name,
                'page': page_idx,
                'panel': panel_idx,
                'path': str(panel_path),
                'bbox': [x1, y1, x2, y2],
        })

     



        print(f'Page {page_idx:04d}: {len(sorted_indices)} panels detected')

    print(f'\nTotal panels extracted: {len(all_panels)}')
    return all_panels


if __name__ == '__main__':
    CBZ_PATH = '/media/hdd/Old Boy/Old Boy Vol. 01 (Dark Horse).cbz'
    MANGA_NAME = 'old_boy_vol01'

    detector = load_detector()
    pages = extract_pages_from_cbz(CBZ_PATH)
    panels = detect_and_crop_panels(detector, pages, MANGA_NAME)

    print(f'\nSample panel entry:')
    print(panels[0])