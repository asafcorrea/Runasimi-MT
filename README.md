# Runasimi-MT: Spanish–Quechua Neural Machine Translation

Fine-tuning a distilled NLLB-200 model for the low-resource **Spanish–Quechua** language pair.

This repository contains the code used for the results reported in the paper *Fine-Tuning a 600M NLLB-200 Model for Low-Resource Spanish-Quechua Machine Translation* (IEEE INTERCON 2026).

---

## Overview

The lack of accessible machine translation tools for Andean languages motivates this work. This repository implements fine-tuning of [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) on the [`somosnlp-hackathon-2022/spanish-to-quechua`](https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua) dataset.

- **Paper:** *Fine-Tuning a 600M NLLB-200 Model for Low-Resource Spanish-Quechua Machine Translation*
- **Authors:** Asaf Reucher Correa Arroyo, Reucher Correa Morocho
- **Venue:** 2026 IEEE XXXIII International Conference on Electronics, Electrical Engineering and Computing (INTERCON), Trujillo, Peru
- **DOI:** [10.1109/INTERCON71992.2026.11768063](https://doi.org/10.1109/INTERCON71992.2026.11768063)

> **Scope note:** The broader research project explores machine translation for Andean languages, including Aymara. This repository currently contains only the Spanish–Quechua code used for the results reported in the paper. Aymara experiments are not included here yet.

---

## 📊 Results

**Test set:** 12,843 examples.

| Source | Checkpoint | BLEU | chrF++ |
|--------|------------|------|--------|
| Paper | step 57,798 (epoch 9) | 17.67 | **42.68** |
| Reproduction | checkpoint-57798 (epoch 9) | 22.89 | 40.87 |
| Reproduction | checkpoint-64220 (epoch 10) | 18.58 | **43.58** |

---

## Reproducibility Notes

Re-running the pipeline gives slightly different results from the paper due to non-determinism (`fp16`, CUDA, library versions) and checkpoint selection. Two checkpoints from the same run were evaluated:

- `checkpoint-64220` is closest to the paper on both metrics (Δ ≈ 0.9), consistent with normal non-determinism.
- `checkpoint-57798` shows a different BLEU/chrF++ trade-off, reflecting the known decoupling between validation loss and generation metrics.
- The default training configuration selects the best checkpoint by **validation loss**, which is `checkpoint-64220` (epoch 10). The paper reported `checkpoint-57798` (epoch 9).

> **Note:** Until full determinism is implemented, the reported numbers should be read as one draw from a distribution rather than a fixed result.

Planned improvements for full reproducibility:

- Fixed random seed (`seed`, `data_seed`, `full_determinism=True`)
- Pinned library versions
- Explicit checkpoint documentation
- Model selection by validation **chrF++** instead of validation loss

---

## ⚙️ Setup

The experiments were run on Google Colab with a single GPU. Main dependencies:

```bash
pip install transformers datasets accelerate sacrebleu sentencepiece
