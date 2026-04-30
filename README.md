# MangaJEPA

A dual-encoder manga comprehension system using V-JEPA 2 and CLIP for panel retrieval, character identification, and grounded question answering.

Built for DA6401W Deep Learning, IIT Madras.

---

## What It Does

MangaJEPA answers questions about manga by retrieving relevant panels and grounding responses in those images — not from web searches or pretraining knowledge. If a panel doesn't exist in the index, the system won't make it up.

Three core features:
- **Q&A**: ask a question, get an answer grounded in retrieved panels
- **Character Explorer**: browse panels by character using V-JEPA 2 prototype retrieval
- **Action Browser**: browse panels by detected action (CLIP zero-shot classification)

Evaluated on Old Boy (Dark Horse, Volumes 1-3) and Manga109.

---

## Key Results

| Experiment | CLIP | V-JEPA 2 |
|---|---|---|
| Manga109 character clustering purity | 0.307 | 0.300 |
| Old Boy clustering purity (145 panels) | 0.744 | 0.731 |
| Old Boy per-character recall (Goto) | 0.726 | **0.915** |
| Old Boy per-character recall (Eri) | 0.833 | **1.000** |
| Label propagation accuracy | **83.3%** | 58.6% |
| Projection layer CLIP agreement | -- | **35.0%** (vs 1.7% baseline) |

**Main finding:** CLIP and V-JEPA 2 have complementary strengths. CLIP is better for text retrieval and label propagation. V-JEPA 2 produces better character clustering and recall when enough seed panels are available.

---

## Setup

### Requirements

- Python 3.11
- NVIDIA GPU (tested on RTX 3060 Ti, 8GB VRAM)
- Ollama with `llama3.2-vision` for local Q&A
- ~15GB disk space for embeddings and models

### 1. Clone and create environment

```bash
git clone https://github.com/harishr-1386/Manga_JEPA.git
cd Manga_JEPA
conda create -n mangajepa python=3.11 -y
conda activate mangajepa
pip install -r requirements.txt
```

### 2. Download V-JEPA 2 weights

```bash
wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -P data/
```

### 3. Configure paths

```bash
cp .env.example .env
# Edit .env with your actual paths
```

Required variables in `.env`:
```
VJEPA2_HUB_PATH=/path/to/.cache/torch/hub/facebookresearch_vjepa2_main
VJEPA2_WEIGHTS=/path/to/data/vitl.pt
PANELS_DIR=/path/to/data/panels
EMBEDDINGS_DIR=/path/to/data/embeddings
MODELS_DIR=/path/to/data/models
PANEL_DETECTOR=/path/to/data/models/manga_panel_detector_fp32.pt
MANGA109_ROOT=/path/to/Manga109_released_2023_12_07
MANGA109_IMAGES=/path/to/Manga109_released_2023_12_07/images
MANGA109_ANNOTATIONS=/path/to/Manga109_released_2023_12_07/annotations
GEMINI_API_KEY=your_key_here  # optional, for Gemini backend
```

### 4. Add manga volumes

Place CBZ files anywhere and update `src/panel_extractor.py` with the path, or use the pipeline script below.

---

## Running the Pipeline

### Option A: Use precomputed embeddings (recommended)

Pre-computed embeddings for Old Boy Vol. 1 are included in `data/embeddings/`. Load directly into ChromaDB:

```bash
python -m src.store
python -m src.app
```

Open `http://localhost:7860`

### Option B: Full pipeline from CBZ

```bash
# 1. Extract panels
python -m src.panel_extractor

# 2. Encode with V-JEPA 2
python -m src.embedder

# 3. Encode with CLIP
python -m src.clip_embedder

# 4. Load into ChromaDB
python -m src.store

# 5. Run action detection
python -m src.action_detector

# 6. Build character index
python -m src.character_retrieval

# 7. Launch app
python -m src.app
```

---

## Running Evaluations

```bash
# Character clustering on Manga109
python -m src.character_clusterer

# In-domain clustering on Old Boy manual labels
python -m src.oldboy_cluster_eval

# Label propagation threshold evaluation (CLIP vs V-JEPA 2)
python -m src.vjepa_threshold_eval

# Retrieval method comparison
python -m src.method_comparison

# LLM model comparison (Llama vs Moondream)
python -m src.eval

# Projection layer training
python -m src.projection_trainer_4k
```

---

## Project Structure

```
src/
  app.py                    # Gradio web application
  panel_extractor.py        # YOLO panel detection and cropping
  embedder.py               # V-JEPA 2 panel encoding
  clip_embedder.py          # CLIP panel encoding
  encoder.py                # V-JEPA 2 model loader
  store.py                  # ChromaDB vector store
  retriever.py              # Panel retrieval functions
  character_retrieval.py    # V-JEPA 2 prototype character retrieval
  action_detector.py        # CLIP zero-shot action classification
  sequential_encoder.py     # V-JEPA 2 sequential panel encoding
  projection_trainer.py     # Projection layer training (997 samples)
  projection_trainer_4k.py  # Projection layer training (4999 samples)
  tagger.py                 # Gradio panel labeling tool (Vol 1)
  tagger_v2.py              # Gradio panel labeling tool (multi-volume)
  manga109_parser.py        # Manga109 XML annotation parser
  character_clusterer.py    # CLIP vs V-JEPA 2 clustering on Manga109
  oldboy_cluster_eval.py    # In-domain clustering evaluation
  label_propagation.py      # CLIP-based label propagation
  threshold_eval.py         # CLIP threshold evaluation
  vjepa_threshold_eval.py   # CLIP vs V-JEPA 2 threshold comparison
  retrieval_eval.py         # Character-aware retrieval evaluation
  method_comparison.py      # Three-way retrieval comparison visualization
  cluster_viz.py            # Clustering visualization
  qa.py                     # LLM grounding layer
  eval.py                   # LLM model evaluation (Llama vs Moondream)

data/
  panels/                   # Cropped panel images
  embeddings/               # V-JEPA 2 and CLIP embeddings (.npy)
  chroma/                   # ChromaDB vector store
  labels/                   # Character labels and character index
  action_labels/            # CLIP action detection results
  models/                   # Downloaded model weights
  cluster_viz/              # Clustering visualization outputs
  eval_grids/               # LLM evaluation grids
  method_comparison/        # Retrieval method comparison grids
  threshold_eval/           # Threshold evaluation charts

docs/
  done.md                   # Project progress log
  issues_faced.md           # Engineering challenges and solutions
```

---

## LLM Backend

The app supports two backends selectable in the UI:

**Ollama (default, free)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2-vision
```

**Gemini Flash (faster, requires API key)**
Get a free API key at `https://aistudio.google.com/apikey` and paste it in the app's LLM Settings panel.

---

## Dataset Attribution

This project uses the Manga109 dataset for evaluation. Please cite:

```bibtex
@article{multimedia_aizawa_2020,
    author={Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa
            and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta},
    title={Building a Manga Dataset ``Manga109'' with Annotations for
           Multimedia Applications},
    journal={IEEE MultiMedia},
    volume={27}, number={2}, pages={8--18},
    doi={10.1109/mmul.2020.2987895},
    year={2020}
}
```

Panel detection uses `leoxs22/manga-panel-detector-yolo26n` trained on Manga109-s.

---

## Future Work

- Fine-tune V-JEPA 2 on Manga109 panel sequences (~50-100 A100 GPU hours)
- Fine-tune CLIP on manga using speech bubble text as captions
- Sequential action detection using V-JEPA 2 temporal embeddings with manual labels
- Automatic character name detection from query text
- Extend character seed panels to minimum 50 per character

---

*IIT Madras | DA6401W Deep Learning | 2026*
