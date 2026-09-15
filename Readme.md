# LLM Student Modeling Strategies

## Steps to Reproduce
1. Set up a Python 3.11 environment with the packages in `requirements.txt`.
2. Put API keys in a `.env` file at the repo root: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY` (only needed for notebooks that call that provider).
3. Download the datasets (see below) into `eedi_data/` and `sciq_data/`.
4. Run the notebooks in ascending numeric order. Several early steps require manual input/labeling (e.g. `002-preprocessing.ipynb` requires labeling problem solvability) — see the notebooks themselves and the paper for details. Notebooks from `010` onward call LLM APIs and are expensive to re-run from scratch; with the original cached responses/results in place they mostly just re-derive statistics.

## Datasets
- **Eedi (math)**: [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data) (Kaggle competition `train.csv`, `misconception_mapping.csv`, into `eedi_data/`). Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- **SciQ (science)**: [SciQ dataset](https://huggingface.co/datasets/allenai/sciq), loaded via `datasets.load_dataset` into `sciq_data/`.

Neither dataset is bundled in this repository.

## Cite
```bibtex
@inproceedings{zengaffinen2026distractors,
  title     = {Can {LLM}s Model Incorrect Student Reasoning? A Case Study on Distractor Generation for Multiple-Choice Questions},
  author    = {Zengaffinen, Yanick and Opedal, Andreas and Rooein, Donya and Srivatsa, Kv Aditya and Sonkar, Shashank and Sachan, Mrinmaya},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```
(Page numbers / ACL Anthology ID to be added once the proceedings are compiled.)

## Contact
Feel free to open an issue or send an email to [yanick.zengaffinen@inf.ethz.ch](mailto:yanick.zengaffinen@inf.ethz.ch).