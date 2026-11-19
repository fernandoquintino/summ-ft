# Prompting, LoRA, QLoRA, or Full Fine-Tuning? Dialogue Summarization with Qwen2.5-0.5B

## Summary
The **objective** of this mini-project was to fine-tune Qwen2.5-0.5B-Instruct to summarize dialogues from SAMSum while measuring forgetting and the cost. My **approach** was to first consider different methods: prompt engineering, LoRA, QLoRA, and full fine-tuning. ROUGE-1 was used to measure a model's ability to summarize. To measure a model's catastrophic forgetting, Massive Multitask Language Understanding (MMLU) was used which measures a model's general knowledge.

The models were compared with the base model (the original Qwen2.5-0.5B-Instruct before fine-tuning): with its best prompt engineering (5-shot ROUGE-1 points) and MMLU points. The highest gains in ROUGE-1 points came from the full fine-tuned model at an increase of 10.5 points, but at a cost of the highest forgetting, a decrease of 10.4 MMLU points. LoRA rank=16 gained 9 ROUGE-1 points from the base model's score, nearly reached the highest gains, while showing no signs of forgetting (gained 0.7 MMLU points).

The models were trained on Kaggle's free tier (2x NVIDIA T4, 16 GB each).

This mini-project extends the labs of the DeepLearning.Ai course Generative AI with Large Language Models to include a current decoder-only model.

## Results
Key findings:
- Fine-tuning beats prompting by 8 - 10.5 ROUGE-1 points on the test set.
- More trainable parameters improve the ROUGE-1 score slightly, but increase forgetting (MMLU) more rapidly.
- LoRA rank=16 had near best ROUGE-1 score and showed no sign of forgetting.
- QLoRA rank=16 saved little memory, but that saving did not outweigh the increase in training time. ROUGE-1 results are 1 point less than LoRA rank=16.
- ROUGE-1/2/L scores on validation set and test set produced similar scores. The order of ROUGE-1 scores on the models was preserved on both the validation set and test set.

| Method     |   ROUGE-1 |   ROUGE-2 |   ROUGE-L |   MMLU | trainable parameters   |   train min | peak memory allocated on GPU (GB) |   eval min |   val loss |
|:-----------|---------:|---------:|---------:|-------:|:-------------------|------------:|:--------------|-----------:|------------:|
| 5-shot     |     37.4 |     12.5 |     28.4 |   41.1 | -               |       -   |       -       |       17.2 |      -      |
| LoRA r=16  |     46.4 |     19.7 |     36.9 |   41.8 | 8798208            |        17.9 | 11.25         |       21.6 |      1.3091 |
| LoRA r=64  |     47.1 |     20.7 |     37.7 |   35.9 | 35192832           |        18   | 11.56         |       21.5 |      1.2978 |
| QLoRA r=16 |     45.4 |     19   |     36.4 |   39.4 | 8798208            |        26.4 | 10.9          |       21.8 |      1.3277 |
| Full FT    |     47.9 |     21.8 |     38.7 |   30.7 | 494032768          |        24.9 | >14.5         |       10.9 |      1.312  |

Notes:
- ROUGE = test set (819 dialogues)
- MMLU = 1,000 fixed questions
- Full FT = checkpoint-900 (see Section What went wrong)
- Metrics from training runs: train min, peak memory allocated (GPU 0), val loss
- eval min measures the time to generate summaries for the test set and compute ROUGE.

## Figures
In the next two figures, ROUGE-1 scores are computed on the validation set and include all the models that were considered.

<img src="figures/rouge_val_vs_params.png" alt="ROUGE-1 vs trainable parameters" width="70%">

The scatter plot illustrates that as the number of trainable parameters increases, the ROUGE-1 score increases at a very slow rate. Differences in ROUGE-1 are small.


<img src="figures/rouge_gain_val_vs_forgetting.png" alt="ROUGE-1 vs forgetting (MMLU)" width="70%">

The plot illustrates that generally the more the model specializes (increases ROUGE-1 score), the more the model forgets its general knowledge (MMLU score decreases). The exceptions are LoRA rank=4 and LoRA rank=16 showing no measurable forgetting (LoRA rank=16 increased slightly).

## Setup
- Model: `Qwen/Qwen2.5-0.5B-Instruct` 494,032,768 parameters.
- Data: `Samsung/samsum` is no longer on Hugging Face Hub. Used `knkarthick/samsum`. Train set has 14,731 samples, validation set has 818 samples, and test set has 819 samples.
- Prompting: Changed system prompt to "Summarize this dialogue in one or two sentences."
- Loss masking: Only the summary and `<|im_end|>` count toward the loss. The prompt and the trailing `\n` are masked, so that the model learns to stop at `<|im_end|>`.
- Lengths: The longest dialogue is 1,049 tokens. Every example was kept and max length set to 1,200 tokens. Each batch was padded only to its longest example (dynamic padding).

## Methods
| | Few-shot | LoRA | QLoRA | Full FT |
|-|-|-|-|-|
| What trains | nothing | A, B adapters | A, B on a 4-bit base | all weights |
| Key settings |0/1/3/5 shots|r = 4, 16, 64, $\alpha$ = 2r, all 7 linear layers|r = 16, NF4, double quantization|gradient checkpointing|
| Learning rate |-|2e-4|2e-4|2e-5|

Shared settings:
- 1 epoch
- effective batch 16 = 2 per GPU x 4 gradient accumulation steps x 2 GPUs (DDP)
- cosine learning-rate scheduler
- 3% warmup
- fp16 mixed precision
- best checkpoint by validation loss

## Quantization
Key Findings:
- Quantization reduced memory footprint by 52% for 4bit (34.3% for 8bit).
- Quantization made the MMLU evaluation slower by 1.5 times for 4bit (2.75 for 8bit) than without quantization. Summary generation and scoring were also slower by similar ratios.
- Changes in quality (ROUGE-1/2/L and MMLU points) were small.

Quantization was applied to the best all-around model, LoRA rank=16, and evaluated on the validation set.

| quant   |   memory footprint GB |   peak_gen_gb |   peak_mmlu_gb |   rouge1 |   rouge2 |   rougeL |   mmlu |   rouge_min |   mmlu_min |
|:--------|---------------:|--------------:|---------------:|---------:|---------:|---------:|-------:|------------:|-----------:|
| none    |           1.02 |          1.14 |           1.5  |     47.1 |     21.3 |     37.7 |   41.7 |        21.9 |        1.2 |
| 8bit    |           0.67 |          0.77 |           1.13 |     47.5 |     21.3 |     37.7 |   40.5 |        59.1 |        3.3 |
| 4bit    |           0.49 |          0.6  |           0.96 |     46.2 |     20   |     36.6 |   40.4 |        30.5 |        1.8 |

Notes:
- `peak_gen_gb` and `peak_mmlu_gb` record the peak memory allocated during the summarizing/ROUGE scoring and MMLU scoring, respectively.
- `rouge_min` is the time needed to summarize the validation set and run the ROUGE scoring.

## What went wrong
- Full fine-tuning ran out of memory at step 920/921. Checkpoint at step 900 was used. At step 900 the learning rate was about 3e-8, so any changes after that were small. The crash showed that training needed more than 14.5 GB of allocated memory. The T4's total capacity is 14.56 GiB (about 15.6 GB), about 0.9 GiB of it was reserved but unused (fragmented).
- A batch size of 4 per GPU ran out of memory, so switched to batch size of 2 per GPU and doubled the gradient accumulation steps to keep the effective batch size at 16.
- In a short test with DataParallel, GPU 0 peaked at about 14.8 GB (vs 11.3 GB with one GPU), close to the T4's limit. I switched to DistributedDataParallel and the full LoRA r=16 run peaked at 11.25 GB (GPU 0) and trained in about 18 min (vs about 33 min estimate for one GPU).

## Limitations
- Single seed per run.
- MMLU on a 1,000-question subset.
- Trained for only 1 epoch.
- QLoRA's memory saving was small in these tests probably due to the model's small size (0.5B parameters).
- Few-shot examples were drawn with one seed.
- ROUGE measures word overlap, not faithfulness.
- No confidence intervals on ROUGE.

## Reproduce
Run all commands from the repository root:
```
pip install -r requirements.txt
pip install -e .
```
`requirements.txt` installs `bitsandbytes` (used for QLoRA training and the quantization results) only on Linux. These runs were done on Kaggle's NVIDIA T4 GPUs.

On Kaggle (GPU T4 x 2), uninstall torchao (the preinstalled version breaks peft):
```
pip uninstall -y torchao
```

Train on 2 GPUs:
```
torchrun --nproc_per_node 2 -m summft.train --method lora --rank 4
torchrun --nproc_per_node 2 -m summft.train --method lora --rank 16
torchrun --nproc_per_node 2 -m summft.train --method lora --rank 64
torchrun --nproc_per_node 2 -m summft.train --method qlora --rank 16
torchrun --nproc_per_node 2 -m summft.train --method full --grad-ckpt
```

Evaluate:
```
python -m summft.run_icl --shots 5 --name base --split test
python -m summft.run_icl --model <model or adapter folder> --name <label> --split test
python -m summft.run_mmlu --model <model or adapter folder> --name <label>
```
Add `--quant 8bit` or `--quant 4bit` for quantized evaluation.

Note: `--name` sets the output folder:
`results/<name>_<shots>shot_<split>/` for ROUGE, `results/mmlu_<name>_1000questions/` for MMLU. The notebook (`explore.ipynb`) expects the following names and settings to generate the images/tables:
 - ROUGE and MMLU:  `base` (no `--model`; few-shot via `--shots`), `lora_r4`, `lora_r16`, `lora_r64`, `qlora_r16`, `full`
 - Quantization runs (on the validation split, the default): `lora_r16_none`, `lora_r16_8bit`, `lora_r16_4bit`

Experiment log with all decisions and intermediate results: [LOG.md](LOG.md)

## Repository structure
| File | Contents |
|-|-|
| `src/summft/data.py`| loads SAMSum, builds chat prompts (zero/few-shot), tokenizes, custom padding|
| `src/summft/train.py`| fine-tunes with LoRA, QLoRA, or full fine-tuning|
| `src/summft/generate.py`| generates summaries |
| `src/summft/metrics.py`| computes ROUGE-1/2/L |
| `src/summft/mmlu.py`| model picks A/B/C/D from MMLU questions and scores|
| `src/summft/models.py`| loads the base model, a LoRA adapter, or a full checkpoint (optionally: 8-bit/4-bit quant).|
| `src/summft/run_icl.py`| generates summaries for a split and saves ROUGE scores|
| `src/summft/run_mmlu.py`| runs the MMLU evaluation and saves score|
|`tests/`| Contains the pytests.|
|`figures/`| Contains the plots.|
|`LOG.md`| The Experiment log.|
|`explore.ipynb`| Generates the tables and plots.|
