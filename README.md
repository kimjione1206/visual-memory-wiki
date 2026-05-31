# Visual Memory Wiki

Visual Memory Wiki is an experimental tool for image-based knowledge exploration.

It turns Markdown notes into visual cards, places those cards in a similarity graph, and exports an HTML viewer plus an Obsidian JSON Canvas. The project is intentionally framed as a research prototype, not a finished personal knowledge management product.

## What It Tests

The core hypothesis:

```text
Markdown notes
→ keyword / visual cards
→ CLIP or TF-IDF coordinates
→ short visual cue retrieval
→ source-grounded knowledge nodes
```

The image is not treated as the fact. The image is a navigation handle. The original Markdown source remains the ground truth.

## Current Finding

In the bundled demo, generated text-free image cards retrieved better than hand-drawn text-free cards when the query was a short visual cue:

```text
Short visual cue only:
- deterministic text-free cards: 33%
- generated image cards: 67%
```

Long task prompts mixed directly into CLIP performed worse. The more promising design is:

```text
task context
→ LLM summarizes it into a short visual cue
→ CLIP retrieves image nodes
→ source text is read and cited
```

This is an early result on a tiny demo set. It should not be read as a general benchmark.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the CLIP backend:

```bash
pip install -e ".[clip]"
```

Without the optional CLIP dependency, the CLI falls back to TF-IDF where possible.

## Quick Start

Build a local visual wiki from the bundled demo notes:

```bash
visual-wiki build --docs demo_knowledge --out dist/demo --backend auto --card-style textless
```

Open:

```text
dist/demo/visual-memory-wiki.html
dist/demo/my-wiki/visual-memory-wiki.canvas
dist/demo/my-wiki/visual-retrieval-walk.md
```

Run the small demo evaluation:

```bash
visual-wiki eval --docs demo_knowledge --out dist/eval
```

Use generated cards from a folder:

```bash
visual-wiki build \
  --docs demo_knowledge \
  --out dist/generated-demo \
  --backend clip \
  --generated-card-dir /path/to/generated_cards
```

Generated card filenames must start with node ids such as:

```text
n01_image-as-link.png
n02_upload.png
```

## Outputs

`visual-wiki build` writes:

- `visual-memory-wiki.html`: browser viewer
- `my-wiki/visual-memory-wiki.canvas`: Obsidian JSON Canvas
- `my-wiki/notes/*.md`: source-grounded node notes
- `my-wiki/visual-retrieval-walk.md`: retrieval trace
- `run_report.json`: machine-readable run metadata

## Why This Is Experimental

Known limitations:

- The demo dataset is tiny.
- Good generated images matter a lot.
- CLIP can be sensitive to prompt phrasing.
- Long task prompts should not be sent directly as visual retrieval queries.
- Generated images can hallucinate. Always keep source text attached.
- The current evaluation is illustrative, not a robust benchmark.

## Repository Layout

```text
visual_memory_wiki/
  cards.py        # deterministic visual card generation
  cli.py          # visual-wiki CLI
  docs.py         # Markdown loading and keyword extraction
  embeddings.py   # TF-IDF and optional CLIP encoders
  exporters.py    # HTML, Obsidian Canvas, Obsidian notes
  graphing.py     # similarity graph and layout
  retrieval.py    # cue ranking and retrieval walk
demo_knowledge/   # tiny demo notes
tests/            # smoke tests
```

The older root-level scripts are kept as experiment logs for now. New usage should go through `visual-wiki`.

## License

MIT. Generated image assets may have their own terms depending on the model/provider used to create them; do not assume this repository's license covers third-party generated assets unless they are explicitly included and licensed.
