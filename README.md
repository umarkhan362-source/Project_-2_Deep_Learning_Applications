# Project-2 Deep Learning Applications

## Project 2, Week 5:  Neural Networks, Convolutions, and Transfer Learning

- **Dataset:** Intel Image Classification, 6 classes (buildings, forest, glacier, mountain, sea, street), 14,034 train (split into 11,929 train + 2,105 validation) / 3,000 test
- **Model:** ResNet-18 pretrained on ImageNet (11,179,590 parameters in total, new 6-class head)
- **Feature extraction (3,078 trainable parameters):** val accuracy 0.905 (3 epochs, only the new head trained)
- **Fine-tuning layer4 (about 8.4M trainable, exactly 8,396,806):** val accuracy 0.933 (3 more epochs, learning rate 1e-4 for layer4 and 1e-3 for the head)
- **Test accuracy (used once):** 0.933 (2,798 of 3,000 images correct); weakest class: glacier (recall 0.881), confused with mountain (47 glaciers predicted as mountain, 49 mountains predicted as glacier)
- **Most instructive error:** a dark city-street photo labelled "buildings" was predicted "street" with confidence 1.00; the scene fits both labels, so the image is genuinely ambiguous rather than the model being plainly wrong.
- **Next:** my own domain in Week 6: apply the same transfer-learning pipeline to my own image dataset (see the proposal below).

### Results in detail

| Stage | Trainable parameters | Best validation accuracy |
|---|---|---|
| Feature extraction (head only) | 3,078 (0.03% of the network) | 0.905 |
| Fine-tuning (layer4 + head) | 8,396,806 (about 75%) | 0.933 |

Fine-tuning improved validation accuracy by 0.028 (2.8 percentage points). Validation loss fell from 0.274 to 0.208, and showed no sign of rising, although the gap to training loss (0.120) widened by the last epoch.

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| buildings | 0.937 | 0.918 | 0.927 | 437 |
| forest | 0.998 | 0.989 | 0.994 | 474 |
| glacier | 0.894 | 0.881 | 0.887 | 553 |
| mountain | 0.901 | 0.884 | 0.892 | 525 |
| sea | 0.947 | 0.982 | 0.964 | 510 |
| street | 0.928 | 0.950 | 0.939 | 501 |
| **Overall accuracy** | | | **0.933** | **3,000** |

Largest confusions: glacier and mountain (47 + 49 = 96 errors), then buildings and street (34 + 24 = 58 errors). My prediction in Task 1.2, that glacier vs mountain would be confused most, was correct. Many of these errors are scenes where two classes legitimately overlap.

### Method

- Images resized to 224 x 224 and normalised with ImageNet mean and standard deviation.
- Augmentation (random resized crop, horizontal flip, mild colour jitter) on the training set only; validation and test use the plain eval transform.
- Validation set (15% of the training folder) used to pick the best epoch and compare strategies; the test set was used once at the end.
- Optimizer: Adam, batch size 64, 3 epochs feature extraction + 3 epochs fine-tuning; the best validation epoch is kept.

### Files

- `week-5-neural-networks.ipynb`: the Kaggle notebook (Parts 1 to 7, with a "Write it" cell after every part and the Check Your Understanding answers)
- `resnet18_intel.pt`: saved weights (44.8 MB, under GitHub's 100 MB limit)

### Reload the model

```python
import torch, torch.nn as nn
from torchvision import models

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load('resnet18_intel.pt', map_location='cpu'))
model.eval()
```

Class order: buildings, forest, glacier, mountain, sea, street. Use the same 224 x 224 + ImageNet normalisation at prediction time.

### Note on the run

This run executed on CPU (the Kaggle GPU was not enabled), so epochs took about 13 to 17 minutes instead of about 1 minute on a T4.

### Week 6 proposal

**To complete before Week 6 (one paragraph):** the problem, who benefits, the dataset link, and the number of classes and images. Example structure: "I will classify [what] from images to help [who]. I plan to use [dataset name and link], which has [n] classes and about [n] labelled images, with a [license] license."
