# Project-2 Deep Learning Applications

**Author:** Muhammad Umar Khan 

**Course:** Introduction to Applied AI (BS 8th Semester, Department of Electronics) 

**Platform:** Kaggle notebook

## Project 2, Week 5:  Neural Networks, Convolutions, and Transfer Learning

- **Dataset:** Intel Image Classification, 6 classes, 14,034 train (11,929 train + 2,105 validation) / 3,000 test
- **Model:** ResNet-18 pretrained on ImageNet
- **Feature extraction (3,078 trainable parameters):** val accuracy 0.905
- **Fine-tuning layer4 (about 8.4M trainable, 8,396,806):** val accuracy 0.933
- **Test accuracy (used once):** 0.933; weakest class: glacier (recall 0.881), confused with mountain
- **Most instructive error:** a true "buildings" photo of a dark city street between tall towers was predicted "street" with confidence 1.00, which suggests the label itself is ambiguous rather than the model being clearly wrong.
- **Next:** my own domain in Week 6 (see the proposal at the end of this file)

---

## 1. Goal

Build a first deep learning image classifier and understand every number it produces: check a convolution by hand, count parameters, fine-tune a pretrained ResNet-18 on six scene classes, and analyse its mistakes. The same pipeline is the template for my own domain in Week 6.

**Classes (label order):** `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

## 2. Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (puneet6060). Images are $150 \times 150 \times 3$ (`uint8`, values 0 to 255), which is $150 \times 150 \times 3 = 67{,}500$ numbers per image.

| Class | Train folder | Test |
|---|---|---|
| buildings | 2,191 | 437 |
| forest | 2,271 | 474 |
| glacier | 2,404 | 553 |
| mountain | 2,512 | 525 |
| sea | 2,274 | 510 |
| street | 2,382 | 501 |
| **Total** | **14,034** | **3,000** |

**Splits.** The training folder is split 85% / 15% with a fixed seed (42):

$$n_{val} = \lfloor 0.15 \times 14{,}034 \rfloor = 2{,}105, \qquad n_{train} = 14{,}034 - 2{,}105 = 11{,}929$$

| Split | Images | Transform |
|---|---|---|
| Train | 11,929 | Augmented (random resized crop, horizontal flip, colour jitter) |
| Validation | 2,105 | Plain resize, no augmentation |
| Test | 3,000 | Plain resize, no augmentation, used once at the end |

**Preprocessing (must match ImageNet pretraining):** resize to $224 \times 224$, `ToTensor` ($x \leftarrow x/255$), then normalise per channel:

$$x_c \leftarrow \frac{x_c - \mu_c}{\sigma_c}, \quad \mu = [0.485, 0.456, 0.406], \; \sigma = [0.229, 0.224, 0.225]$$

## 3. Method

### 3.1 Convolution and parameter counts (by hand, then checked in PyTorch)

$$\text{output size} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1 \qquad\qquad \text{conv parameters} = (K \cdot K \cdot C_{in} + 1) \cdot C_{out}$$

| Check | Calculation | Result |
|---|---|---|
| Edge kernel on the lecture $5 \times 5$ image | $(-1)(0) + (0)(0) + (1)(10)$ per row, 3 rows | 30 (output `[30, 30, 0]`) |
| Conv $3 \times 3$, 3 to 64 channels | $(3 \cdot 3 \cdot 3 + 1) \cdot 64$ | **1,792** parameters |
| Dense $67{,}500 \to 512$ | $67{,}500 \cdot 512 + 512$ | **34,560,512** parameters |
| Dense vs conv | $34{,}560{,}512 / 1{,}792$ | about **19,286 times** more |
| Stride 1, $224 \times 224$ input | $\frac{224 - 3 + 2}{1} + 1$ | 224 |
| Stride 2, $224 \times 224$ input | $\lfloor \frac{224 - 3 + 2}{2} \rfloor + 1$ | 112 |

A conv layer's size does not depend on the image size (weight sharing: one filter reused everywhere), and the same filter detects a pattern wherever it appears.

### 3.2 Training loop

Every iteration runs the same four steps:

| Step | Code | Math |
|---|---|---|
| Forward | `logits = model(xb)` | $z = f_\theta(x)$ |
| Loss | `F.cross_entropy(logits, yb)` | $p_k = \dfrac{e^{z_k}}{\sum_j e^{z_j}}, \quad L = -\log p_{true}$ |
| Backward | `loss.backward()` | $\dfrac{\partial L}{\partial w} = \dfrac{\partial L}{\partial \hat y}\cdot\dfrac{\partial \hat y}{\partial z}\cdot\dfrac{\partial z}{\partial w}$ |
| Step | `optimizer.step()` | $w \leftarrow w - \alpha \nabla_w L$ (Adam, per-weight step sizes) |

Batch size 64 ($\lceil 11{,}929 / 64 \rceil = 187$ steps per epoch). `fit()` keeps the weights from the best validation epoch.

### 3.3 Two-phase transfer learning

| Phase | What is trained | Trainable parameters | Learning rate | Epochs |
|---|---|---|---|---|
| Feature extraction | New head `Linear(512, 6)` only | $512 \times 6 + 6 = $ **3,078** (0.03% of 11,179,590) | $10^{-3}$ | 3 |
| Fine-tuning | `layer4` + head | $8{,}393{,}728 + 3{,}078 = $ **8,396,806** (75.1%) | $10^{-4}$ (`layer4`), $10^{-3}$ (head) | 3 |

The pretrained block gets a learning rate ten times smaller than the new head, so its good weights are adjusted instead of destroyed (catastrophic forgetting).

## 4. Results

### 4.1 Learning curves (all 6 epochs)

| Epoch | Phase | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|---|
| 1 | Feature extraction | 0.601 | 0.807 | 0.331 | 0.895 |
| 2 | Feature extraction | 0.339 | 0.885 | 0.309 | 0.895 |
| 3 | Feature extraction | 0.308 | 0.892 | 0.274 | **0.905** |
| 4 | Fine-tuning | 0.260 | 0.906 | 0.213 | 0.926 |
| 5 | Fine-tuning | 0.166 | 0.940 | 0.214 | 0.932 |
| 6 | Fine-tuning | 0.120 | 0.956 | 0.208 | **0.933** |

Fine-tuning improved validation accuracy by $0.933 - 0.905 = +0.028$ (2.8 percentage points). Validation loss did not rise, but the gap between training loss (0.120) and validation loss (0.208) is widening, so overfitting is only beginning. Training accuracy is below validation accuracy in the early epochs because training images are augmented (harder) and the training figure is averaged while the weights are still changing.

### 4.2 Test results (test set used once)

**Test accuracy: 0.933** ($2{,}798 / 3{,}000$).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| buildings | 0.937 | 0.918 | 0.927 | 437 |
| forest | 0.998 | 0.989 | 0.994 | 474 |
| glacier | 0.894 | **0.881** | 0.887 | 553 |
| mountain | 0.901 | **0.884** | 0.892 | 525 |
| sea | 0.947 | 0.982 | 0.964 | 510 |
| street | 0.928 | 0.950 | 0.939 | 501 |
| **Macro average** | 0.934 | 0.934 | 0.934 | 3,000 |

$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

**Confusion matrix** (rows = true class, columns = predicted):

| true \ pred | buildings | forest | glacier | mountain | sea | street |
|---|---|---|---|---|---|---|
| buildings | **401** | 0 | 0 | 1 | 1 | 34 |
| forest | 0 | **469** | 4 | 1 | 0 | 0 |
| glacier | 1 | 1 | **487** | 47 | 16 | 1 |
| mountain | 1 | 0 | 49 | **464** | 10 | 1 |
| sea | 1 | 0 | 5 | 2 | **501** | 1 |
| street | 24 | 0 | 0 | 0 | 1 | **476** |

Worked example: glacier recall $= 487 / 553 = 0.881$.

### 4.3 Error analysis

- **Weakest classes by recall:** glacier (0.881) and mountain (0.884).
- **Most confused pair:** glacier and mountain, with $47 + 49 = 96$ errors. The second is buildings and street, with $34 + 24 = 58$ errors. Together these two pairs account for 154 of the 202 errors.
- **My prediction from before training** (glacier vs mountain) was correct.
- **Error gallery** (first 8 errors, 7 of them buildings predicted as street): several are scenes where two labels legitimately overlap, such as a dark city street between tall buildings (predicted street, confidence 1.00) or a stone house in a hilly landscape (predicted mountain, 0.69). One image of an industrial hall with tracks has a questionable label. Many remaining errors are therefore ambiguous or arguably mislabelled images, not clear model failures. The gallery shows errors in dataset order, so it is not a random sample.
- **Softmax check on one test image** (true class: buildings): logits $[7.21, -8.42, -4.16, -4.74, -5.10, -1.86]$ give

$$p_{buildings} = \frac{e^{7.21}}{\sum_j e^{z_j}} = \frac{1352.9}{1353.1} = 0.9999$$

## 5. Model file

`resnet18_intel.pt` holds the `state_dict` only (44.8 MB, under GitHub's 100 MB limit). Size check: $11{,}179{,}590 \times 4 \text{ bytes} \approx 44.7$ MB.

```python
import torch, torch.nn as nn
from torchvision import models

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)      # must match the saved head
model.load_state_dict(torch.load('resnet18_intel.pt', map_location='cpu'))
model.eval()
```

Use the same evaluation transform (resize to 224 x 224, ImageNet normalisation) and the class order `['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']` at prediction time.

## 6. Reproduce

1. Create a Kaggle notebook and add the Intel Image Classification dataset.
2. Enable a GPU accelerator (T4) and Internet, or add ResNet-18 weights (`resnet18-f37072fd.pth`) as an input.
3. Run `week-5-neural-networks.ipynb` from top to bottom.

**Note:** the run reported here was done on CPU (each epoch took about 760 to 1,040 seconds). On a GPU T4 an epoch takes about one minute.

## 7. Week 6 proposal

> **TODO (replace before Week 6):** one paragraph with the problem, who benefits, the dataset link, and the number of classes and images. Candidate domains from the lab: crop disease, medical images, manufacturing defects, documents, satellite images, or my own idea. The dataset should have at least a few hundred labelled images per class and a checked license.
