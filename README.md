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

## Workflow

1. Edit code in `src/dlproject/` in VS Code.
2. `git commit` and `git push`.
3. In Colab, open the notebook (badge in the notebook, or
   `colab.research.google.com/github/tbloesch/deepLearning/blob/main/notebooks/final_project.ipynb`)
   and run the setup cell. It clones/pulls the repo, so you always run the latest code.

## Local setup (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
