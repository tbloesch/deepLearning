# deepLearning

Image classification project. Code is written in VS Code, training runs in Google Colab.

## Structure

```
deepLearning/
├── src/dlproject/        # all reusable code (edit this in VS Code)
│   ├── config.py         # constants: paths, image size, normalisation, batch size
│   ├── data.py           # download dataset, build datasets and dataloaders
│   ├── utils.py          # small helpers (denormalise, list_images, ...)
│   └── checks.py         # data sanity checks (samples, balance, duplicates, ...)
├── notebooks/            # thin notebooks that import from src/ and run things
│   └── final_project.ipynb
├── data/                 # downloaded at runtime, not in git
└── pyproject.toml
```

## My workflow
Create the model in models.py
Create the train.py based on our code from Lab4, specifically the evaluate function and training loop that starts right after
Because we have a small dataset, we only unfreeze 1 layer

After first run - we look at val loss because it gives more information than val accuracy. Can see at epoch 9 the loss is at it's lowest.

Implement saving the model while we train, so I can get the right weights

Experiment 1: Try to do cyclic learning rate
Experiment 2: Data augmentation

Received the first graphs of training val and train val. Starts overfitting after epoch 8 ish. 



## Local setup (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
