import os
import base64
import time
from pathlib import Path
from dotenv import load_dotenv
import ollama
from src.retriever import retrieve_panels

load_dotenv()


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


def ask(manga_name: str, query: str, n_panels: int = 5) -> dict:
    print(f'\nRetrieving panels for: "{query}"')
    panels = retrieve_panels(manga_name, query, n_results=n_panels)

    print(f'Retrieved {len(panels)} panels:')
    for p in panels:
        print(f"  page {p['page']:04d} panel {p['panel']:04d} "
              f"(similarity: {p['similarity']:.3f})")

    # Build images list for Ollama
    images = []
    for panel in panels:
        with open(panel['path'], 'rb') as f:
            images.append(base64.b64encode(f.read()).decode('utf-8'))

    print('\nGenerating grounded response...')
    response = ollama.chat(
        model='moondream',
        messages=[{
            'role': 'user',
            'content': build_prompt(query, panels),
            'images': images,
        }]
    )

    return {
        'query': query,
        'answer': response['message']['content'],
        'panels': panels,
    }


if __name__ == '__main__':
    test_queries = [
        "what is happening in the fighting scenes?",
        "describe the mood and atmosphere of the dark scenes",
        "what are characters doing in the eating scenes?",
    ]

    for query in test_queries:
        result = ask('old_boy_vol01', query)
        print(f'\n{"="*60}')
        print(f'Q: {result["query"]}')
        print(f'A: {result["answer"]}')
        print('='*60)
        time.sleep(5)