import os
import xml.etree.ElementTree as ET
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MANGA109_IMAGES      = Path(os.getenv('MANGA109_IMAGES'))
MANGA109_ANNOTATIONS = Path(os.getenv('MANGA109_ANNOTATIONS'))


def list_titles() -> list[str]:
    """Return all available manga titles."""
    return sorted([
        f.stem for f in MANGA109_ANNOTATIONS.glob('*.xml')
    ])


def parse_annotations(title: str) -> dict:
    """
    Parse a Manga109 XML annotation file.
    Returns a dict with characters, pages, frames, faces, bodies.
    """
    xml_path = MANGA109_ANNOTATIONS / f'{title}.xml'
    tree     = ET.parse(xml_path)
    root     = tree.getroot()

    # Parse character registry
    characters = {}
    for char in root.find('characters'):
        characters[char.attrib['id']] = char.attrib['name']

    # Parse pages
    pages = []
    for page in root.find('pages'):
        page_data = {
            'index':  int(page.attrib['index']),
            'width':  int(page.attrib['width']),
            'height': int(page.attrib['height']),
            'frames': [],
            'faces':  [],
            'bodies': [],
            'texts':  [],
        }

        for elem in page:
            bbox = {
                'id':   elem.attrib.get('id'),
                'xmin': int(elem.attrib['xmin']),
                'ymin': int(elem.attrib['ymin']),
                'xmax': int(elem.attrib['xmax']),
                'ymax': int(elem.attrib['ymax']),
            }
            if elem.tag == 'frame':
                page_data['frames'].append(bbox)
            elif elem.tag == 'face':
                bbox['character_id']   = elem.attrib.get('character')
                bbox['character_name'] = characters.get(
                    elem.attrib.get('character'), 'unknown'
                )
                page_data['faces'].append(bbox)
            elif elem.tag == 'body':
                bbox['character_id']   = elem.attrib.get('character')
                bbox['character_name'] = characters.get(
                    elem.attrib.get('character'), 'unknown'
                )
                page_data['bodies'].append(bbox)
            elif elem.tag == 'text':
                bbox['content'] = elem.text or ''
                page_data['texts'].append(bbox)

        pages.append(page_data)

    return {
        'title':      title,
        'characters': characters,
        'pages':      pages,
    }


def get_image_path(title: str, page_index: int) -> Path:
    """Return path to a specific page image."""
    return MANGA109_IMAGES / title / f'{page_index:03d}.jpg'


def get_all_faces(annotations: dict) -> list[dict]:
    """
    Flatten all face annotations across all pages.
    Returns list of face dicts with page index included.
    """
    faces = []
    for page in annotations['pages']:
        for face in page['faces']:
            faces.append({
                **face,
                'page_index': page['index'],
                'title':      annotations['title'],
                'image_path': str(get_image_path(
                    annotations['title'], page['index']
                )),
            })
    return faces


def get_all_frames(annotations: dict) -> list[dict]:
    """
    Flatten all frame (panel) annotations across all pages.
    """
    frames = []
    for page in annotations['pages']:
        for i, frame in enumerate(page['frames']):
            frames.append({
                **frame,
                'page_index':  page['index'],
                'frame_index': i,
                'title':       annotations['title'],
                'image_path':  str(get_image_path(
                    annotations['title'], page['index']
                )),
            })
    return frames


if __name__ == '__main__':
    titles = list_titles()
    print(f'Total titles: {len(titles)}')
    print(f'First 5: {titles[:5]}')

    # Parse one title as a test
    title = 'ARMS'
    ann   = parse_annotations(title)

    print(f'\nTitle: {ann["title"]}')
    print(f'Characters: {len(ann["characters"])}')
    for cid, name in list(ann["characters"].items())[:5]:
        print(f'  {cid}: {name}')

    print(f'Pages: {len(ann["pages"])}')

    faces  = get_all_faces(ann)
    frames = get_all_frames(ann)
    print(f'Total faces:  {len(faces)}')
    print(f'Total frames: {len(frames)}')
    print(f'\nSample face: {faces[0]}')
    print(f'Sample frame: {frames[0]}')



    