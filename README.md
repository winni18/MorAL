# MorAL

## Setup

```bash
# Create & activate the environment
conda env create -f environment.yml
conda activate calm

# Download spaCy model
python -m spacy download en_core_web_sm

# Install Jericho 
pip install git+https://github.com/jens321/jericho.git@iclr
```

## Models

### CALM 

* **Download:** [Google Drive Link](https://drive.google.com/file/d/1dkjuc_3xY5O0ANKr2QgHz1CoJQuqXYPf/view)
* **Placement:** Save under `model_weights/` (e.g., `model_weights/gpt2/`).
* **Usage:** Set the `--lm_path` argument to point to the model directory.

### CCLM 

* **Repo:** [https://github.com/hendrycks/ethics](https://github.com/hendrycks/ethics)
* **Placement:** Save under `ethics/`.
* **Usage:** Set the `--cclm_path` argument to the checkpoint directory.

## Training

```bash
cd experiments/moral_calm
bash train_ft.sh
```

## Acknowledgements

We thank [Jiminy Cricket benchmark](https://github.com/hendrycks/jiminy-cricket) for providing annotated games and moral evaluation tasks.
