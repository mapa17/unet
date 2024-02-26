# unet
Experiments with UNET

Inspired by https://www.youtube.com/watch?v=IHq1t7NxS8k

## Instructions

### Training
Train a model with data in `./data` storing the model weights in `./WD/`

```bash
python src/main.py -w ./WD train ./data
```

### Run a model
Run inference using the model weights `WD/model_best.pt` on the data in `./data/test/`

```bash
python src/main.py run ./WD/model_best.pt ./data/test/
```