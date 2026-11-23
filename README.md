# deepLearning

Image classification project. Code is written in VS Code, training runs in Google Colab.

## Structure

```
deepLearning/
├── src/dlproject/        # all reusable code (edit this in VS Code)
│   ├── config.py         # constants: paths, image size, normalisation, batch size
│   ├── data.py           # download dataset, build datasets and dataloaders
│   ├── models.py         # build_model: pretrained ResNet-18/50, frozen backbone, optional layer4 unfreeze
│   ├── train.py          # training loop + evaluate, saves the best epoch (by val loss)
│   ├── utils.py          # small helpers (set_seed, denormalise, list_images, ...)
│   └── checks.py         # data sanity checks (samples, balance, duplicates, ...)
├── notebooks/            # thin notebooks that import from src/ and run things
│   └── final_project.ipynb
├── results/              # training histories (JSON, one file per configuration, 3 seeds each)
├── data/                 # downloaded at runtime, not in git
└── pyproject.toml
```

## My workflow
Create the model in models.py
Create the train.py based on our code from Lab4, specifically the evaluate function and training loop that starts right after
Because we have a small dataset, we start with the whole backbone frozen and only train the new last layer (fc). Unfreezing layer4 comes later (Exp 3).

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

Ran flip, jitter and all (flip + crop + jitter) too. None of them beat no augmentation

- All val differences are within seed noise. Val is only 134 images, so 1 image = 0.75 pp.
- Crop and all lower the train acc (harder to memorise), but val doesn't get better. "All" even has a slightly higher val loss (about 2.5 std).

- My explanation: the ImageNet V2 weights were already trained with flips/crops/colour changes, so the frozen features are already invariant to them. And only fc can learn.Crop/all converge more slowly, so 20 epochs may be short for them. Write this as a limitation.
- I skip testing pairs of augmentations, since "all" just behaves like "crop plus a bit more".

Reran "none" to get its histories for the comparison plot. It gave exactly the same numbers as Exp 1b (to 4 decimals), so the seeds make the runs reproducible.

Made a comparison plot of val loss for all 5 configurations (full run + zoom on epochs 10-20). All curves overlap; crop and all are slightly higher in the middle of training.

Since the frozen features seem to be the reason augmentation doesn't help, I unfreeze layer4 so the features can change. Plan is a 2x2: {frozen, layer4} x {none, all}.
- Added `unfreeze_layer4=False` to `build_model`. It first freezes everything, then unfreezes layer4. Checked: frozen ResNet-50 trains 14,343 of 23,522,375 parameters; with layer4 unfrozen it's 14,979,079 (63.7%), so about 1000x more.
- layer4 already has good ImageNet weights, so it needs a lower learning rate than fc (e.g. 1e-4 vs 1e-3) using Adam parameter groups in train.py.

Metrics: val loss picks the best epoch, val acc is the main number during experiments. For the final evaluation on test I also need macro F1 (classes are imbalanced, 3.4x), per-breed precision/recall/F1 and a confusion matrix. mAP is for detection/ranking, not single-label classification.

Plan from here (priority order):
1. Exp 3 (2x2), then pick the final model on val.
2. Test set: load it, group the noise triplets, duplicate check train/val <-> test.
3. Final evaluation once on test: overall, per breed, per noise level, per photo. This answers the second half of our research question (noise robustness).
4. Grad-CAM + examples of misclassified images. Related work incl. Atabay 2017.
5. Only if there's time: Exp 4 (class-weighted loss).


## Local setup (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
