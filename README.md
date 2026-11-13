# Prompting, LoRA, QLoRA, or Full Fine-Tuning? Dialogue Summarization with Qwen2.5-0.5B

## Summary
The **objective** of this mini-project was to fine-tune Qwen2.5-0.5B-Instruct to summarize dialogues from SAMSum while measuring forgetting and the cost. My **approach** was to first consider different methods: prompt engineering, LoRA, QLoRA, and full fin-tuning. ROUGE-1 was used to measure a model's ability to summarize and to measure a model's general forgetting Massive Multitask Language Understanding (MMLU) was used which measures a model's general knowledge.

The highest gains in ROUGE-1 points from prompt engineering came from the full fine tuned model at 10.5 points, but at a cost of the highest forgetting, a decrease of 10.4 MMLU points from the baseline (the model's MMLU score without training). LoRA rank=16 gained 9 ROUGE-1 points from the baseline, nearly reached the highest gains, while showing no signs of forgetting (gained 0.7 MMLU points).

The models where trained on Kaggle's free tier (2x NVIDIA T4, 16 GB each)

## Results
Key findings:
- Fine tuning beats prompting by 8 - 10.5 ROUGE-1 points on the test set.
- More trainable parameters improves the ROUGE-1 score slightly, but increases forgetting (MMLU) more rapidly.
- LoRA rank=16 had near best ROUGE-1 score and showed no sign of forgetting.
- QLoRA rank=16 saved little memory, but not out weight the increase in training time. ROUGE-1 results are 1 point less than LoRA rank=16.
- ROUGE-1/2/L scores on validation set and test set produced similar scores. The order of ROUGE-1 scores on the models was preserved on both the validation set and test set.

| Method     |   Rouge 1 |   Rouge 2 |   Rouge L |   MMLU | trainable parameters   |   train min | peak memory allocated on gpu (GB) |   eval min |   val loss |
|:-----------|---------:|---------:|---------:|-------:|:-------------------|------------:|:--------------|-----------:|------------:|
| 5-shot     |     37.4 |     12.5 |     28.4 |   41.1 | <NA>               |       nan   |               |       17.2 |             |
| LoRA r=16  |     46.4 |     19.7 |     36.9 |   41.8 | 8798208            |        17.9 | 11.25         |       21.6 |      1.3091 |
| LoRA r=64  |     47.1 |     20.7 |     37.7 |   35.9 | 35192832           |        18   | 11.56         |       21.5 |      1.2978 |
| QLoRA r=16 |     45.4 |     19   |     36.4 |   39.4 | 8798208            |        26.4 | 10.9          |       21.8 |      1.3277 |
| Full FT    |     47.9 |     21.8 |     38.7 |   30.7 | 494032768          |        24.9 | >14.5         |       10.9 |      1.312  |
Notes:
- ROUGE = test set (819 dialogues)
- MMLU = 1,000 fixed questions
- Full FT = checkpoint-900 (see Section What went wrong)
- Metrics from training runs: train min, peak memory allocated, val loss
- eval min measures the time to generate Rouge summaries from the test set and evaluate Rouge scores.

## Figures
In the next two figures, ROUGE-1 scores are computed on the validation set and include all the models that were considered.

<img src="figures/rouge_val_vs_params.png" alt="ROUGE-1 vs trainable parameters" width="70%">

The scatter plot illustrates that as the trainable parameters increases, the ROUGE-1 score increases at a very slow rate. Differences in ROUGE-1 are insignificant.


<img src="figures/rouge_gain_val_vs_forgetting.png" alt="ROUGE-1 vs forgetting (MMLU)" width="70%">

The plot illustrates that generally the more the model specializes (increases ROUGE-1 score), the more the model forgets its general knowledge (MMLU score decreases). The exception is LoRA rank=4 and LoRA rank=16 showing no significant forgetting (LoRA rank=16 increased slightly).
## Quantization
Key Findings:
- Quantization reduced model footprint by 52% for 4bit (34.3% for 8bit).
- Quantization was slower by 1.5 times for 4bit (2.75 for 8bit) than without quantization.


| quant   |   model footprint GB |   peak_gen_gb |   peak_mmlu_gb |   rouge1 |   rouge2 |   rougeL |   mmlu |   rouge_min |   mmlu_min |
|:--------|---------------:|--------------:|---------------:|---------:|---------:|---------:|-------:|------------:|-----------:|
| none    |           1.02 |          1.14 |           1.5  |     47.1 |     21.3 |     37.7 |   41.7 |        21.9 |        1.2 |
| 8bit    |           0.67 |          0.77 |           1.13 |     47.5 |     21.3 |     37.7 |   40.5 |        59.1 |        3.3 |
| 4bit    |           0.49 |          0.6  |           0.96 |     46.2 |     20   |     36.6 |   40.4 |        30.5 |        1.8 |
Notes:
- `peak_gen_gb` and `peak_mmlu_gb` record the peak memory allocated during the summarizing/ROUGE-1/2/L testing and MMLU testing, respectfully.
- `rouge_min` is the time needed to summarize the validation set and run the ROUGE-1/2/L tests.

## What went wrong
## Limitations
## Reproduce
## Repository layout