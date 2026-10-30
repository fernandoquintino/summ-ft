## 2026-10-24 — Setup
- Did: repo, venv, pytest, Kaggle clone via GitHub token
- Result: smoke test passes locally and on Kaggle
- Fix: local Python was 3.14, Kaggle 3.13.15 -> rebuilt venv with 3.13.17
- Fix: `import summft` failed -> added pyproject.toml + `pip install -e .`
- Pinned: transformers 5.16.1, datasets 4.8.5, pytest 8.4.2 (Kaggle versions)

## 2026-10-24 — Data lengths
- Dataset: `Samsung/samsum` no longer on Hub -> using `knkarthick/samsum`
  (14,731 train; official is 14,732)
- Tokenizer: Qwen2.5-0.5B-Instruct
- Dialogue tokens: mean 140.8, median 114, 95th 342, 99th 490, max 1049
- Summary tokens: mean 25.8, median 23, 95th 54, 99th 67, max 86
- Decision: keep all examples, cap 1,200, dynamic padding.
  Fallback: drop longest 1% of train if full fine-tune runs out of memory

## 2026-10-27 - loss masking
- Mask trailing `\n` at end of assistant summary so that model learns to stop at `<|im_end|>`.

## 2026-10-28 - system message
- Change system content to "Summarize this dialogue in one or two sentences."

## 2026-10-29 - review of in context learning
- In context learning on full validation set:
```  shots  rouge1  rouge2  rougeL  minutes   n
     0    33.8    10.0    25.2     18.4 818
     1    35.0    11.4    27.7     10.9 818
     3    38.4    13.2    29.7     15.9 818
     5    38.9    13.8    29.8     18.0 818```
- lengths of output:
```  0 shots: pred len = 33.2 ref len = 20.3
  1 shots: pred len = 18.5 ref len = 20.3
  3 shots: pred len = 26.3 ref len = 20.3
  5 shots: pred len = 29.3 ref len = 20.3```
-reading a few examples showed that the difference was minimal, just reworded
-Conclusion: few shots helped. 0 shots took longer probably because it had the longest output

## 2026-10-30 - MMLU baseline
- 1,000 fixed questions, Kaggle 2xT4 fp16:
  41.1 % 0.7 min