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

Implement saving the model while we train, so I can get the right weights.

Gives 4 percentage difference than first run -> Try to implement seed instead
It's because I have shuffle=true in my data.py and nn.linear is also random.

Now we have 3 different runs, but they all top at 10 epochs.
Let's try to run the same model for 20 epochs, to make sure it's not too short

Okay now I run it with both 18 and 50 to make sure. 
I've collected the results and 50 was best. 

Now I implemented some data augmentation. I want to test different ones singular first, and then compare them at the end.

I start with crop - but it made almost no difference. Maybe because my backbone is frozen, since that results in my features not being able to change. Only the last linear layer changes. 
I'm gonna run the flip and jitter as well, just to make sure. 



## Local setup (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
