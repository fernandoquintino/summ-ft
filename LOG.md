## 2026-10-24 — Setup
- Did: repo, venv, pytest, Kaggle clone via GitHub token
- Result: smoke test passes locally and on Kaggle
- Fix: local Python was 3.14, Kaggle 3.13.15 -> rebuilt venv with 3.13.17
- Fix: `import summft` failed -> added pyproject.toml + `pip install -e .`
- Pinned: transformers 5.16.1, datasets 4.8.5, pytest 8.4.2 (Kaggle versions)

## 2026-10-24 — Data lengths
- Dataset: `Samsung/samsum` no longer on Hub → using `knkarthick/samsum`
  (14,731 train; official is 14,732)
- Tokenizer: Qwen2.5-0.5B-Instruct
- Dialogue tokens: mean 140.8, median 114, 95th 342, 99th 490, max 1049
- Summary tokens: mean 25.8, median 23, 95th 54, 99th 67, max 86
- Decision: keep all examples, cap 1,200, dynamic padding.
  Fallback: drop longest 1% of train if full fine-tune runs out of memory