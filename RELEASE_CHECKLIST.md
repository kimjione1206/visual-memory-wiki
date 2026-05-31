# Release Checklist

This repository is currently suitable to publish as an honest research prototype:

> image-based knowledge exploration experiment for Markdown notes.

Before pushing to GitHub:

- Keep generated experiment outputs out of the repository. They are ignored by `.gitignore`.
- Include only small demo Markdown notes under `demo_knowledge/`.
- Do not claim this is a finished LLM wiki or production RAG system.
- State that images are navigation handles, not ground truth.
- State that generated image assets may have separate provider/model terms.
- Run the smoke checks below.

## Smoke Checks

```bash
python3 -m py_compile visual_memory_wiki/*.py
python3 -m visual_memory_wiki.cli --help
python3 -m visual_memory_wiki.cli build --docs demo_knowledge --out dist/demo-tfidf --backend tfidf --card-style textless
python3 -m visual_memory_wiki.cli eval --docs demo_knowledge --out dist/eval-tfidf --no-clip
```

Optional CLIP checks:

```bash
python3 -m visual_memory_wiki.cli build --docs demo_knowledge --out dist/demo-clip --backend auto --card-style textless
python3 -m visual_memory_wiki.cli eval --docs demo_knowledge --out dist/eval-clip
```

Optional generated-card check:

```bash
python3 -m visual_memory_wiki.cli eval \
  --docs demo_knowledge \
  --out dist/eval-generated \
  --generated-card-dir generated_gpt_image_eval/gpt_cards
```

## Current Local Verification

On the local machine used for the prototype:

- `python3 -m py_compile visual_memory_wiki/*.py`: passed
- Manual core smoke test: passed
- `visual-wiki build` equivalent via `python3 -m visual_memory_wiki.cli build`: passed for TF-IDF and CLIP
- `visual-wiki eval` equivalent via `python3 -m visual_memory_wiki.cli eval`: passed for TF-IDF, CLIP, and generated cards
- `pytest`: not run here because `pytest` is not installed in the active Python environment; it is listed under the `dev` optional dependency.

Observed demo metric with generated cards:

```text
generated_card_clip accuracy@1 = 0.6667
generated_card_clip MRR = 0.75
```

This is a tiny illustrative demo, not a benchmark.
