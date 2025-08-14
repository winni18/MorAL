# MorAL

## Setup

```bash
# Create & activate env
conda env create -f environment.yml
conda activate calm

# spaCy model
python -m spacy download en_core_web_sm

# Jericho (3.x)
pip install git+https://github.com/jens321/jericho.git@iclr
```

## Models

**Pretrained CALM (GPT-2)**
Download: [https://drive.google.com/file/d/1dkjuc\_3xY5O0ANKr2QgHz1CoJQuqXYPf/view](https://drive.google.com/file/d/1dkjuc_3xY5O0ANKr2QgHz1CoJQuqXYPf/view)
Place under: `model_weights/` (e.g., `model_weights/gpt2/`), and set `--lm_path` accordingly.

**CCLM (RoBERTa ethics, optional)**
Repo: [https://github.com/hendrycks/ethics](https://github.com/hendrycks/ethics)
Place under: `ethics/` (e.g., `ethics/roberta-large/`), and set `--cclm_path` to the checkpoint dir.

## Train

Use `train_ft.sh` or run directly:

## References

* **Jericho** (text-adventure interface)
* **ETHICS** (RoBERTa moral judgements)
* **Jiminy Cricket** (annotated games & moral evaluation tasks)
