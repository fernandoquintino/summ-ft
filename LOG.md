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
```
  shots  rouge1  rouge2  rougeL  minutes   n
     0    33.8    10.0    25.2     18.4 818
     1    35.0    11.4    27.7     10.9 818
     3    38.4    13.2    29.7     15.9 818
     5    38.9    13.8    29.8     18.0 818
```
- lengths of output:
```
  0 shots: pred len = 33.2 ref len = 20.3
  1 shots: pred len = 18.5 ref len = 20.3
  3 shots: pred len = 26.3 ref len = 20.3
  5 shots: pred len = 29.3 ref len = 20.3
```
-reading a few examples showed that the difference was minimal, just reworded
-Conclusion: few shots helped. 0 shots took longer probably because it had the longest output

## 2026-10-30 - MMLU baseline
- 1,000 fixed questions, Kaggle 2xT4 fp16:
  41.1 % 0.7 min

## 2026-11-06 - LoRA results (validation, 1 epoch, alpha = 2r, all linear layers)
```
    method  rouge1  rouge2  rougeL  mmlu  trainable_params  train_min  peak_mem_gb  eval_min
   0-shot    33.8    10.0    25.2  41.1              <NA>        NaN          NaN      18.4
   1-shot    35.0    11.4    27.7  41.1              <NA>        NaN          NaN      10.9
   3-shot    38.4    13.2    29.7  41.1              <NA>        NaN          NaN      15.9
   5-shot    38.9    13.8    29.8  41.1              <NA>        NaN          NaN      18.0
 LoRA r=4    47.1    20.6    37.3  40.6           2199552       18.6        11.17      20.1
LoRA r=16    47.2    21.3    37.6  41.8           8798208       17.9        11.25      19.9
LoRA r=64    47.6    21.8    38.2  35.9          35192832       18.0        11.56      21.0
```

## 2026-11-07 - Full fine tuning
```
  method  rouge1  rouge2  rougeL  mmlu  trainable_params  train_min peak_mem_gb  eval_min  eval_loss
   0-shot    33.8    10.0    25.2  41.1              <NA>        NaN        None      18.4        NaN
   1-shot    35.0    11.4    27.7  41.1              <NA>        NaN        None      10.9        NaN
   3-shot    38.4    13.2    29.7  41.1              <NA>        NaN        None      15.9        NaN
   5-shot    38.9    13.8    29.8  41.1              <NA>        NaN        None      18.0        NaN
 LoRA r=4    47.1    20.6    37.3  40.6           2199552       18.6       11.17      20.1     1.3303
LoRA r=16    47.2    21.3    37.6  41.8           8798208       17.9       11.25      19.9     1.3091
LoRA r=64    47.6    21.8    38.2  35.9          35192832       18.0       11.56      21.0     1.2978
  Full FT    48.8    22.5    39.1  30.7         494032768       24.9       >14.5      11.0     1.312
```
- Trade off: Slightly better rouge metrics (1 - 2 points), mmlu score is the worse 5 - 6 points from nearest LoRA (and 11 points from baseline).
- Full FT ran out of memory at 920/921 (GPU 1). Used checkingpoint-900 as the final model (eval_loss: 1.312, learning_rate: 2.994e-08)
- Peak memory not recorded. Crash showed it needed >14.5 GB (T4 limit)
- Full FT's eval_loss is higher than LoRA r=16,64 but has better Rouge metrics. eval_loss and rouge don't equate exactly.

## 2026-11-11 - QLoRA results
```
    method  rouge1  rouge2  rougeL  mmlu  trainable_params  train_min peak_mem_gb  eval_min eval_loss
    0-shot    33.8    10.0    25.2  41.1              <NA>        NaN        None      18.4      None
    1-shot    35.0    11.4    27.7  41.1              <NA>        NaN        None      10.9      None
    3-shot    38.4    13.2    29.7  41.1              <NA>        NaN        None      15.9      None
    5-shot    38.9    13.8    29.8  41.1              <NA>        NaN        None      18.0      None
  LoRA r=4    47.1    20.6    37.3  40.6           2199552       18.6       11.17      20.1    1.3303
 LoRA r=16    47.2    21.3    37.6  41.8           8798208       17.9       11.25      19.9    1.3091
 LoRA r=64    47.6    21.8    38.2  35.9          35192832       18.0       11.56      21.0    1.2978
QLoRA r=16    46.5    20.0    36.7  39.4           8798208       26.4        10.9      21.2    1.3277
   Full FT    48.8    22.5    39.1  30.7         494032768       24.9       >14.5      11.0     1.312
```
- QLoRA r=16 vs Lora r=16: longer training for QLoRA (26.4 min vs 17.9 min); slightly worse Rouge1/2/L scores (46.5, 20.0, 36.7 vs 47.2, 21.3, 37.6); slightly worse mmlu (39.4 vs 41.8); the advantage of QLoRA should have been peak_mem_gb, but the advantage was small (10.9 GB vs 11.25) probably due to the model's small .5B parameters.

## 2026-11-12 - Quantization results
```
quant  model_mem_gb  peak_gen_gb  peak_mmlu_gb  rouge1  rouge2  rougeL  mmlu  rouge_min  mmlu_min
 none          1.02         1.14          1.50    47.1    21.3    37.7  41.7       21.9       1.2
 8bit          0.67         0.77          1.13    47.5    21.3    37.7  40.5       59.1       3.3
 4bit          0.49         0.60          0.96    46.2    20.0    36.6  40.4       30.5       1.8
 ```
 - Quantization reduced the memory foot print and the peak memory allocated ont he gpu.
 - Scores stayed about the same.
 - The cost of quantization was time: no quant 21.9 rouge_m (1.2 mmlu_min) vs 8bit 59.1 rouge_min (1.2 mmlu_mmin) and 4bit 30.5 rouge_min (1.8 mmlu_min).