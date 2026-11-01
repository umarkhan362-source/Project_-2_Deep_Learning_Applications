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
 
**Domain: medical images (brain MRI).** Reading brain MRI scans is a standard medical-imaging task, and the Week 5 pipeline (pretrained ResNet-18, feature extraction, then fine-tuning) can be reused on it with a grayscale setting.
 
**Problem.** A brain tumour has to be found on an MRI scan and its type identified, which takes a trained specialist and a lot of time per scan. I will build an image classifier that takes one brain MRI slice and predicts one of 4 classes: **glioma, meningioma, pituitary tumour, or no tumour**. The model is meant as a screening and second-look aid for a clinician, not as a diagnosis.
 
**Who benefits.** Radiologists and clinicians, who could use it to flag scans that need a closer look sooner, and students learning medical image analysis. Because a missed tumour costs far more than a false alarm, I will report recall (sensitivity) for each tumour class, not only overall accuracy.
 
**Dataset.** The Brain Tumor MRI Dataset on Kaggle (Masoud Nickparvar): **7,023 brain MRI images in 4 classes** (glioma, meningioma, no tumor, pituitary). It combines three source datasets, and the "no tumor" images come from the Br35H dataset. Link: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>. I will check the licence on the Kaggle page before publishing anything and use the data for this course project only.
 
**Plan.**
- Use `GRAYSCALE = True` (MRI slices are grayscale, and the template repeats the channel to 3 for the pretrained network), resize to $224 \times 224$, and normalise with the ImageNet mean and standard deviation.
- The image sizes differ and some images have extra margins, so I will inspect the sizes first and crop the borders if it is clearly needed before resizing.
- If the dataset ships with separate training and testing folders, I will keep the testing folder untouched as my test set and use a grouped split of the training folder for validation (no group in both train and validation). If it does not, I will carve a grouped test set from the data.
- Check for exact and near-duplicate images between the splits before training, because the dataset was assembled from several sources.
- Train a new 4-class head first (`Linear(512, 4)`, $512 \times 4 + 4 = 2{,}052$ trainable parameters), then fine-tune `layer4` with two learning rates ($10^{-3}$ head, $10^{-4}$ `layer4`), use class weights if the classes turn out to be uneven, keep the best validation epoch by macro-F1, and evaluate on the test set once.
- Report macro-F1, per-class precision and recall, a confusion matrix and an error gallery, and use Grad-CAM to see whether the model looks at the tumour region.
**Expected challenges.**
- **Possible shortcut from the data source.** The "no tumor" class comes from a different source dataset than the tumour classes, so the model could learn to recognise the source (image style, margins, intensity) instead of the tumour. I will test this with Grad-CAM and by looking at the errors.
- **Label quality.** The dataset's author states that one source's glioma images were mislabelled and were replaced with images from another site. I will watch for confusion between tumour classes and treat suspicious errors as a possible label problem in my error analysis.
- **Possible patient overlap.** If several slices come from the same patient and the file names do not show a patient ID, a random split can place the same patient in both train and test and make the score too high. I will check the file names for any patient or source ID, and say so in the README if the groups cannot be recovered.
- **Not a clinical tool.** A high score on one public dataset does not show the model is safe for real patients, and I will state this when I report the results.


## Week 6: Brain-Tumour MRI Screening (No Tumour vs Tumour) with ResNet-18

**Author:** Muhammad Umar Khan | **Course:** Introduction to Applied AI | **Date:** 24 October 2026

**Who benefits:** radiologists and clinicians in an MRI reporting workflow, and the patients waiting in their queue. **Why it matters:** a missed tumour (false negative) delays treatment, while a false alarm costs extra review time. The model is meant to prioritise scans for a radiologist, so the metric that matters most is sensitivity at a stated specificity, not accuracy.

**Dataset:** Brain Tumor MRI Dataset (Kaggle, https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset); 7,200 images indexed, 4 folders (glioma, meningioma, pituitary, notumor) merged into k = 2 classes (`no_tumor`, `tumor`); labels come from the dataset's own folder names. The labelling process, the licence and the source scanners are **not documented in the notebook** and should be confirmed on the Kaggle dataset page before any publication.

**Data audit:** imbalance ratio 3.0; exact duplicates 340 files in 153 groups (187 redundant copies removed); test images with a near-duplicate in train 230 of 974 (23.6%), validation 260 of 1,169 (22.2%), all removed from validation and test; split by image file name (no patient id exists), 0 shared groups.

**Imbalance handling:** class-weighted cross-entropy (`USE_SAMPLER = False`), weights no_tumor = 2.087 and tumor = 0.658.

**Results (test, used once):** macro-F1 0.955; weakest class no_tumor (recall 0.877); for binary: sensitivity 0.972 and specificity 0.965 at threshold 0.986 (set on val).

**Grad-CAM:** the model looks at the central brain, ventricles and skull-base region; shortcut risk: no text or corner marker was highlighted in the 6 heat-maps checked, but two error cases show heat near the image edge (see Section 8).

**Limitations:** one public Kaggle dataset, unknown scanners and population, no patient ids, small healthy test group (57 images). Not for diagnosis, treatment decisions or use without radiologist review.

---

## 1. Problem and Setup

| Item | Value |
|---|---|
| Task | Binary classification of brain MRI slices: `no_tumor` vs `tumor` (positive class = `tumor`) |
| Backbone | ImageNet-pretrained ResNet-18 (`resnet18-f37072fd.pth`), new `Linear(512, 2)` head |
| Input | Grey-scale converted to 3 channels, 224 x 224, ImageNet mean/std normalisation |
| Trainable parameters, phase 1 | 1,026 = 512 x 2 + 2 |
| Training | Phase 1: head only, 2 epochs, lr 1e-3. Phase 2: head + `layer4`, 3 epochs, lr 1e-4 (block) and 1e-3 (head), Adam |
| Model selection | Best epoch by validation macro-F1 (not accuracy) |
| Target | Recall of `tumor` at least 0.95 (`TARGET_RECALL`) |
| Hardware | CPU only; about 350 to 460 s per epoch |

---

## 2. Datasheet

| Section | Answer |
|---|---|
| **Motivation** | Public brain-MRI dataset for tumour-type classification. Here it is reused for a screening-style task: is a tumour present? |
| **Composition** | 7,200 images: glioma 1,800, meningioma 1,800, pituitary 1,800, notumor 1,800. After merging: tumor 5,400, no_tumor 1,800. File names such as `Te-gl_1.jpg` and `Tr-pi_984.jpg` carry **no patient id**. Known duplicates: 340 exact-duplicate files. |
| **Collection** | In a 300-image sample, width ranges 173 to 1,338 px (median 512), height 183 to 1,304 px (median 512); colour modes RGB 168, L 131, RGBA 1. Scanner, site, time period and population are not stated in the notebook. |
| **Labelling** | Labels are taken from the folder names. Who labelled them and how disagreements were handled is not documented here (to confirm from the dataset page). |
| **Uses** | Suitable for teaching and for prototyping triage models. Should NOT be used for clinical diagnosis, for children or other populations not represented, or on scans from other scanners without external validation. |

---

## 3. Data Audit

### 3.1 Class balance

$$
\text{imbalance ratio}=\frac{n_{\max}}{n_{\min}}=\frac{5400}{1800}=3.0,\qquad
\text{tumour share}=\frac{5400}{7200}=75\%
$$

Images per group = 7,200 / 7,200 = 1.00, so every image is its own group and a patient-level split is impossible.

### 3.2 Duplicates

| Check | Result |
|---|---|
| Exact duplicate files (MD5) | 340 files in 153 groups, so 340 - 153 = **187** redundant copies removed |
| Images after de-duplication | 7,200 - 187 = **7,013** |
| Near-duplicates in train (fingerprint distance below 0.10) | Test **230 of 974 = 23.6%**; validation **260 of 1,169 = 22.2%** |

Fingerprint distance (32 x 32 standardised thumbnail, 1,024 pixels):

$$
z=\frac{a-\bar a}{\sigma_a+10^{-6}},\qquad
d(z_1,z_2)=\frac{1}{1024}\sum_{i=1}^{1024}\left|z_{1,i}-z_{2,i}\right|,\qquad d<0.10\Rightarrow\text{near-duplicate}
$$

The closest test distances printed were all 0, i.e. identical thumbnails. Near-duplicates were removed from validation and test only (never from train).

### 3.3 Split (StratifiedGroupKFold, 6 folds, seed 42)

| Split | Before near-dup removal | After removal | tumor | no_tumor |
|---|---|---|---|---|
| Train | 4,870 | 4,870 | 3,703 | 1,167 |
| Validation | 1,169 | 909 | 840 | 69 |
| Test | 974 | 744 | 687 | 57 |

$$
\frac{7013}{6}\approx1169\ (\text{val}),\qquad 7013-1169=5844,\qquad \frac{5844}{6}=974\ (\text{test}),\qquad 5844-974=4870\ (\text{train})
$$

Groups shared by train and validation: **0**; by train and test: **0**. Because each image is its own group, this is guaranteed by construction and does **not** prove there is no patient leakage; it is why the content-based near-duplicate check was needed. Three-quarters of the healthy validation images (211 of 280 = 75.4%) had a near-twin in train, which shows how much the no_tumor score would have been inflated.

**Prevalence shift:** tumour is 3703 / 4870 = 76.0% of train but 687 / 744 = 92.3% of test, so PPV and NPV on this test set do not describe a screening population.

---

## 4. Imbalance Handling
Class weight and weighted cross-entropy:
 
$$
w_c=\frac{N}{K\cdot n_c},\qquad
\mathcal{L}=-\frac{\sum_i w_{y_i}\log p_{i,y_i}}{\sum_i w_{y_i}}
$$
 
With N = 4,870 and K = 2:
 
$$
w_{\text{no-tumor}}=\frac{4870}{2\times1167}=\frac{4870}{2334}=2.087
$$
 
$$
w_{\text{tumor}}=\frac{4870}{2\times3703}=\frac{4870}{7406}=0.658
$$
 
Check: 1,167 x 2.087 = 3,703 x 0.658 = 2,435 = N / K, so both classes contribute equally to the loss. A healthy image counts about 3.2 times as much as a tumour image.
 
**Augmentation kept:** random resized crop (scale 0.8 to 1.0), horizontal and vertical flips, rotation up to 20 degrees, brightness and contrast jitter of 0.15. Flips and rotations are used because MRI slices come in many orientations (axial, sagittal, coronal) and the head position varies. Hue and saturation jitter is excluded because the scans are grey-scale. Evaluation uses a plain resize only.
 
---


 


## 5. Training and Model Selection

$$
F1_c=\frac{2P_cR_c}{P_c+R_c},\qquad \text{macro-F1}=\frac{1}{K}\sum_{c=1}^{K}F1_c
$$

| Phase | Epoch | Validation macro-F1 | Time per epoch |
|---|---|---|---|
| 1 (head only) | 1 | 0.903 | 348 s |
| 1 (head only) | 2 | 0.846 | 360 s |
| 2 (fine-tune `layer4`) | 1 | **0.924** (best) | 459 s |
| 2 (fine-tune `layer4`) | 2 | 0.907 | 454 s |
| 2 (fine-tune `layer4`) | 3 | 0.913 | 458 s |

Fine-tuning the last block raised the best validation macro-F1 from 0.903 to 0.924. Scores fluctuate between epochs because the validation set holds only 69 healthy images.

---

## 6. Test Results (used once)

### 6.1 Default decision rule (argmax)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| no_tumor | 0.962 | 0.877 | 0.917 | 57 |
| tumor | 0.990 | 0.997 | 0.993 | 687 |

Confusion matrix (rows = true, columns = predicted): no_tumor 50 / 7, tumor 2 / 685.

$$
\text{macro-F1}=\frac{0.917+0.993}{2}=0.955,\qquad
\text{weighted-F1}=\frac{57(0.917)+687(0.993)}{744}\approx0.988,\qquad
\text{accuracy}=\frac{50+685}{744}=0.988
$$

Accuracy and weighted-F1 (0.988) are dominated by the tumour class. Macro-F1 (0.955) exposes the **weakest class: no_tumor**, where 7 of 57 healthy scans (12.3%) were flagged as tumour.

### 6.2 Recall-driven threshold (set on validation)

The threshold is the highest value giving validation recall of at least 0.95: **t = 0.986** (flag tumour only if $p_{\text{tumor}}\ge0.986$). Applied once to the test set:

| TP | FN | TN | FP | AUC |
|---|---|---|---|---|
| 668 | 19 | 55 | 2 | 0.994 |

$$
\text{Sensitivity}=\frac{TP}{TP+FN}=\frac{668}{687}=0.972,\qquad
\text{Specificity}=\frac{TN}{TN+FP}=\frac{55}{57}=0.965
$$

Test recall (0.972) **reached the 0.95 target**. Claim to report as a pair: *sensitivity 0.972 at specificity 0.965*.

### 6.3 PPV at two prevalences

$$
\text{PPV}=\frac{\text{sens}\cdot\text{prev}}{\text{sens}\cdot\text{prev}+(1-\text{spec})(1-\text{prev})}
$$

| Setting | Prevalence | Working | PPV |
|---|---|---|---|
| Test set | 0.923 | $\dfrac{0.9723\times0.9234}{0.9723\times0.9234+0.0351\times0.0766}=\dfrac{0.8978}{0.9005}$ | **0.997** |
| Realistic screening (5%, as suggested in the lab) | 0.05 | $\dfrac{0.9723\times0.05}{0.9723\times0.05+0.0351\times0.95}=\dfrac{0.0486}{0.0486+0.0333}=\dfrac{0.0486}{0.0820}$ | **0.593** |

In plain language: the same model with the same sensitivity and specificity is almost always right when it says "tumour" in this test set, because 92% of the test scans really are tumours. In a screening population where only 5% have a tumour, healthy scans are 19 times more common, so even a 3.5% false-positive rate produces many false alarms and only about 6 in 10 positive flags are real. (At 2% prevalence PPV would fall to about 0.36.) The test-set PPV of 0.997 must therefore not be quoted as the screening performance.

---

## 7. Imbalance and Metric Choice Summary

| Decision | Choice | Reason |
|---|---|---|
| Loss | Weighted cross-entropy ($w_c=N/(Kn_c)$) | Corrects the 3:1 class imbalance without changing batches |
| Selection metric | Macro-F1 on validation | Gives the small healthy class equal importance; accuracy would hide it |
| Operating point | Threshold 0.986 from validation ROC | Meets the 0.95 recall requirement for the costly error (missed tumour) |
| Reported metrics | Macro-F1, per-class recall, sensitivity, specificity, PPV at two prevalences | Prevalence changes PPV |

---

## 8. Grad-CAM Shortcut Check

$$
\alpha_k^c=\frac{1}{Z}\sum_i\sum_j\frac{\partial y^c}{\partial A^k_{ij}},\qquad
L^c_{\text{Grad-CAM}}=\text{ReLU}\!\left(\sum_k\alpha_k^c A^k\right)
$$

Computed on the last block (`layer4`, 512 maps of 7 x 7), up-sampled to 224 x 224. Six test heat-maps (3 correct, 3 wrong):

| # | Case (true / predicted) | Where the model looked | Legitimate evidence? |
|---|---|---|---|
| 1 | tumor / tumor (glioma) | Central brain, ventricles, skull base | Yes, inside the head |
| 2 | tumor / tumor (glioma) | Central brain around the ventricles | Yes, but diffuse |
| 3 | tumor / tumor (glioma) | Central brain, ventricles | Yes, but diffuse |
| 4 | tumor / no_tumor (glioma, missed) | Left image border, away from the visible lesion; scan is zoomed and cropped | No, looks at the edge, not the lesion |
| 5 | no_tumor / tumor (false alarm) | Right brain surface close to the skull | Weak, near the skull and image edge |
| 6 | no_tumor / tumor (false alarm) | Skull base and brainstem region of a low axial slice | Plausible but anatomical, not a lesion |

The balanced 8-panel figure (one correct case per original folder plus errors) shows the same pattern: for correct tumour cases the heat sits on the lesion or the central and skull-base region (for example the meningioma mass and the sella for pituitary), and the missed pituitary and glioma cases have diffuse or off-target heat.

**Verdict: shortcut suspected (low confidence), not confirmed.** No text, corner marker or label was highlighted, which is the failure shown in the lecture's corner-marker example. However, two error cases attend to the image edge or skull, and Grad-CAM is coarse (7 x 7), so it detects shortcuts but does not localise disease. **Follow-up experiment:** mask or crop the outer border and skull, re-score the 744 test images, and check whether the 19 false negatives and 7 false positives change; also compare performance by original folder.

---

## 9. Responsible Domain AI (four required answers)

| Question | Answer for this model |
|---|---|
| **Who could be harmed, and how?** | A patient with a tumour who is told "no tumour" (false negative) could have treatment delayed: 19 of 687 tumour scans (2.8%) were missed at threshold 0.986. A healthy patient flagged as tumour (false positive) suffers anxiety and unnecessary follow-up scans: 7 of 57 healthy scans (12.3%) at argmax, 2 of 57 (3.5%) at the chosen threshold. |
| **Where was it NOT validated?** | Other hospitals and scanners, other patient populations (age, region, including Pakistani patients), real screening prevalence, other tumour types, and entire patients held out (no patient ids exist, so patient-level splitting was impossible). Only 57 healthy test images were available. |
| **What is the human role?** | Triage only: prioritise a radiologist's queue. A radiologist makes every final diagnosis; the model never reports a result on its own. |
| **What evidence would change my conclusions?** | External validation on scans from a local hospital with patient-level splits and PPV and NPV at the local prevalence; a larger and more balanced healthy test set; a masking experiment showing performance does not depend on the border or skull; and radiologist review of the Grad-CAM maps and of the 26 errors (19 FN + 7 FP). |

---

## 10. Reproducibility and Files

- Seeds: `torch.manual_seed(42)`, `np.random.seed(42)`, `StratifiedGroupKFold(random_state=42)`.
- Saved checkpoint: `week6_model.pt` (`state_dict`, `classes = ['no_tumor', 'tumor']`, `backbone = 'resnet18'`, `img_size = 224`, `grayscale = True`).
- The operating threshold **0.986 is not stored in the checkpoint** and must be applied separately at prediction time.
- Notebook: https://github.com/umarkhan362-source/Project_-2_Deep_Learning_Applications/blob/main/week-6-brain-tumour-mri.ipynb.


## Week 7: Text and Multimodal AI

Project 2 (Deep Learning Applications), individual. Domain: brain-tumour MRI (my Week 6 ResNet-18 model).
Notebook: `week-7-text-multimodal-ai.ipynb` (Kaggle, CPU). Every number below is copied from that notebook's output.

### Summary (the lab's README template, filled in with measured values)

- **Text classifier:** SMS Spam Collection (Almeida et al. 2011), 5,572 texts (747 spam, 13.4%), macro-F1 **0.974** (word 1-2 grams) vs **0.982** (character 2-5 grams)
- **Top positive / negative features:** call, txt, free, text, reply, uk / my, me, ok, that, it, but
- **NER on my domain:** out-of-the-box F1 **0.722** -> with entity ruler **1.000** on 18 held-out entities (provided lab sentences: 0.533 -> 0.875)
- **Regex fields extracted:** CNIC, mobile, amount, date, e-mail, invoice no., scan file, tumour size, MRN, age, MRI sequence; **false positives tested:** 5 negative tests, 8/8 domain tests pass
- **OCR:** CER raw **0.000** -> after correction rules **0.000** on the synthetic invoice (lab pairs: 0.062 / 0.059 / 0.143 -> 0.000)
- **Pipeline:** Week 6 image model + text evidence -> masked JSON record (example below)
- **Privacy:** CNIC, phone, e-mail and MRN masked before extraction and logging; the output JSON was searched for leaked digits and none were found

---

### 1. TF-IDF by hand

Smooth IDF and length-1 (L2) normalisation, as scikit-learn computes them:

$$\text{idf}(t)=\ln\!\left(\frac{1+N}{1+\text{df}(t)}\right)+1 \qquad \mathbf v_d=\frac{\text{tf}\cdot\text{idf}}{\lVert \text{tf}\cdot\text{idf}\rVert_2} \qquad \cos(a,b)=\mathbf v_a\cdot\mathbf v_b$$

Three messages: d1 = "win free cash now", d2 = "call me now", d3 = "free call now" (N = 3).

| Word | df | idf = ln(4/(1+df)) + 1 | d1 weight (by hand) | d1 weight (scikit-learn) |
|---|---|---|---|---|
| win | 1 | ln 2 + 1 = 1.6931 | 1.6931 / 2.8969 = 0.584 | 0.584 |
| free | 2 | ln(4/3) + 1 = 1.2877 | 1.2877 / 2.8969 = 0.445 | 0.445 |
| cash | 1 | ln 2 + 1 = 1.6931 | 0.584 | 0.584 |
| now | 3 | ln 1 + 1 = 1.0000 | 1.0000 / 2.8969 = 0.345 | 0.345 |

$$\lVert \mathbf w_{d_1}\rVert=\sqrt{1.6931^2+1.2877^2+1.6931^2+1.0^2}=\sqrt{8.3918}=2.8969$$

"now" gets the lowest weight because it appears in every message, so it cannot tell messages apart. The notebook asserts that the hand values equal the library values.

$$\cos(d_2,d_3)=\underbrace{0.548\times0.620}_{\text{call}}+\underbrace{0.425\times0.481}_{\text{now}}=0.340+0.204=\mathbf{0.544}\quad(\text{slide 10: }0.54)$$

---

### 2. Text classification: word vs character n-grams

Pipeline: `TfidfVectorizer` -> `LogisticRegression(class_weight='balanced')`, 80/20 stratified split (`random_state=42`): 4,457 training and 1,115 test messages (966 ham, 149 spam).

$$P(\text{spam}\mid x)=\sigma(\mathbf w\cdot\mathbf x+b),\;\;\sigma(z)=\frac1{1+e^{-z}} \qquad \text{tf}'=1+\ln(\text{tf}) \qquad c_k=\frac{n_{\text{samples}}}{K\,n_k}$$

"Always ham" already scores $4825/5572=0.866$ accuracy, so accuracy is not the metric: I report macro-F1 and spam precision/recall.

| Model | Features | Macro-F1 | Spam precision | Spam recall | Spam F1 | Errors |
|---|---|---|---|---|---|---|
| word 1-2 grams (`min_df=2`, sublinear tf) | 12,128 | **0.974** | 0.979 | 0.933 | 0.955 | 13 |
| char_wb 2-5 grams (`min_df=2`, sublinear tf) | 36,449 | **0.982** | 0.986 | 0.953 | 0.969 | 9 |

Confusion matrices recovered from the reports:

| | Word: pred ham | Word: pred spam | Char: pred ham | Char: pred spam |
|---|---|---|---|---|
| **true ham (966)** | 963 | 3 | 964 | 2 |
| **true spam (149)** | 10 | 139 | 7 | 142 |

$$P_{\text{spam}}^{\text{word}}=\frac{139}{139+3}=0.979,\quad R_{\text{spam}}^{\text{word}}=\frac{139}{149}=0.933,\quad \text{macro-}F_1=\frac{0.993+0.955}{2}=0.974$$

$$P_{\text{spam}}^{\text{char}}=\frac{142}{144}=0.986,\quad R_{\text{spam}}^{\text{char}}=\frac{142}{149}=0.953,\quad \text{macro-}F_1=\frac{0.995+0.969}{2}=0.982$$

The character model is ahead by 0.008 macro-F1, which is only 4 of 1,115 messages, so on clean English SMS I treat the models as roughly equal. The case for characters is robustness, and the price is $36{,}449/12{,}128=3.0\times$ more features and weights that are fragments, not words.

**Top features (word model).**

| Direction | n-grams | My reading |
|---|---|---|
| Spam-like | call, txt, free, text, reply, uk, to, www, mobile, claim, stop, your | mostly the classic SMS-spam vocabulary; "to" and "your" are function words (shortcuts) |
| Ham-like | my, me, ok, that, it, but, ll, gt, lt, can, what, da | chat words; "gt" and "lt" are leftovers of `&gt;` / `&lt;`, "da" is an Indian-English chat particle |

The shortcut has a cost: the ham message "Waiting for your call." scores $p=0.85$ ($z=\ln\frac{0.85}{0.15}=1.74$) because "your" and "call" are both top spam features.

**Error analysis (word model, 13 errors = 10 false negatives + 3 false positives), classified by hand:**

| Category | Count | Share | Example |
|---|---|---|---|
| model error | 8 | 61.5% | "ringtoneking 84484" (p = 0.19); "Waiting for your call." (p = 0.85) |
| genuinely ambiguous | 4 | 30.8% | a warning about a scam that quotes the scam (p = 0.14) |
| label error | 1 | 7.7% | "Do you realize that in about 40 years..." labelled spam (p = 0.09) |

**Tricky messages** (flagged as spam when $p\ge0.5$):

| Message | Expected | Word p(spam) | Char p(spam) |
|---|---|---|---|
| W1N a FR33 pr1ze!! txt CLAIM now | spam | 0.99 | 0.85 |
| kal milte hain, 5 baje | ham | 0.19 | 0.12 |
| C0ngrats!! U w0n Rs 25000 pr1ze. Cl@im by txt 5588 now (mine) | spam | 0.81 | 0.88 |
| Mubarak ho! Aap ne 50000 ka inaam jeeta hai, abhi 8000 par SMS karein (mine, Roman Urdu) | spam | **0.32 (missed)** | 0.53 (barely over) |
| Yaar meeting kal subah 9 baje hai, notes bhej dena (mine, Roman Urdu) | ham | 0.13 | 0.12 |

Neither model was trained on Roman Urdu, so the 0.53 is untrained behaviour, not proof of support.

---

### 3. Named-entity recognition, measured

An entity counts as correct only if **both its exact text and its label** match the gold annotation:

$$P=\frac{TP}{TP+FP}\qquad R=\frac{TP}{TP+FN}\qquad F_1=\frac{2PR}{P+R}$$

**Provided lab sentences (6 sentences, 16 gold entities)**

| Pipeline | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| spaCy `en_core_web_sm` out of the box | 8 | 6 | 8 | 8/14 = 0.571 | 8/16 = 0.500 | **0.533** |
| + 7 entity-ruler patterns | 14 | 2 | 2 | 14/16 = 0.875 | 14/16 = 0.875 | **0.875** |

Out-of-the-box failures: "Engro Fertilizers" labelled PERSON, "30/09/2026" labelled GPE, a mobile number labelled DATE, "PKR 45,000" not found, "250,000" labelled CARDINAL.

**My domain: 10 brain-tumour MRI report sentences (names, amounts and dates invented).** Rules were written from sentences 1-5 only; sentences 6-10 are held out.

| Set | Gold entities | Pipeline | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 1-5 (rule-writing) | 20 | spaCy alone | 10 | 6 | 10 | 0.625 | 0.500 | 0.556 |
| 1-5 (rule-writing) | 20 | + domain rules | 20 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| **6-10 (held out)** | 18 | spaCy alone | 13 | 5 | 5 | 0.722 | 0.722 | **0.722** |
| **6-10 (held out)** | 18 | + domain rules | 18 | 0 | 0 | 1.000 | 1.000 | **1.000** |

$$F_1^{\text{held-out, before}}=\frac{2(0.722)(0.722)}{0.722+0.722}=0.722\qquad F_1^{\text{held-out, after}}=\frac{2(1.000)(1.000)}{1.000+1.000}=1.000$$

What the rules actually fixed on the held-out set (all five are fixed formats):

| spaCy alone said | Gold | Fixed by |
|---|---|---|
| `05/12/2026` as ORG | DATE | dd/mm/yyyy date pattern |
| `PKR 52,500` as EVENT | MONEY | money pattern |
| `80,000` as CARDINAL | `Rs. 80,000` MONEY | money pattern |
| `Hassan Raza PKR` as PERSON | `Hassan Raza` PERSON | money pattern |
| `24,000` as CARDINAL | `PKR 24,000` MONEY | money pattern |

**Honest reading.** 13 of 18 held-out entities (all four hospitals, the cities, "9 January 2027", four of five people) were already correct before any rule, so this set barely tests my hospital-suffix pattern or city list. One wrong entity moves F1 by about $1/18=0.056$, and the held-out sentences share their formats with the first five. The result shows that **format rules carry over to new sentences of the same style**, not that they carry over to unseen styles.

**What I would use for what is left:** closed vocabularies (hospital names, cities) -> rules or a gazetteer; rigid formats (amounts, dd/mm/yyyy) -> regex or ruler; open-class person names and org-versus-person cases (e.g. a hospital named after a person) -> a model trained on my own annotated data, which needs far more than 38 gold entities.

---

### 4. Regex fields with negative tests

| Field | Pattern | Example match |
|---|---|---|
| CNIC | `\b\d{5}-\d{7}-\d\b` | 35202-1234567-1 |
| Mobile | `(?:\+92[- ]?\|\b0)3\d{2}[- ]?\d{7}\b` | 0300-1234567, +92 300 1234567 |
| Amount | `\b(?:PKR\|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)` | Rs. 12,500.50 -> 12,500.50 |
| Date | `\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b` | 30/09/2026 |
| E-mail | `\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b` | info@engro.com.pk |
| Invoice no. | `\b(?:Invoice\s+)?No\.?\s*(\d{3,})\b` | Invoice No. 4417 -> 4417 |
| Scan file (mine) | `\b(?:Tr\|Te)-(?:gl\|me\|no\|pi)_\d+\.(?:jpe?g\|png)\b` | Te-gl_1.jpg |
| Tumour size (mine) | `\b\d+(?:\.\d+)?\s?(?:mm\|cm)\b` | 2.5 cm |
| MRN (mine) | `\bMRN[- ]?\d{6,8}\b` | MRN 4471823 |
| Age (mine) | `\baged?\s+(\d{1,2}\|1[01]\d)\b` | aged 46 -> 46 |
| MRI sequence (mine) | `(?-i:\b(?:T1\|T2\|FLAIR\|DWI)\b)` | T2, FLAIR (case-sensitive) |

| Test suite | Positive | Negative | Passed |
|---|---|---|---|
| Lab tests | 3 | 2 (14-digit order number is not a CNIC; "Room 0300" is not a mobile) | 5 / 5 |
| My domain tests | 3 | **5** | **8 / 8 = 100%** |

My five negative tests: `BP 120 mmHg and glucose 5 mmol/L` (not a size), `She has a flair for painting` (not the FLAIR sequence), `Tx-gl_5.jpg and Te-xx_9.jpg` (wrong prefix / class code), `MRN 12 was seen in age 2026` (too few digits; not an age), `Doctors 500 mg twice daily` (not an amount).

**A pattern that matched too much.** My first tumour-size pattern `\d+\s?mm` returned `['120 mm']` on "BP 120 mmHg" (the "mm" of mmHg). Adding `\b` at both ends fixed it, because a boundary after "mm" fails when "Hg" follows. The lab's `amount` pattern had the same flaw at the front: without `\b` before "Rs", "Doctors 500 mg" returned `['500']`. After the fix both return `[]`.

---

### 5. Character error rate and OCR repair

$$\text{CER}=\frac{S+D+I}{N}\quad(\text{Levenshtein edits} / \text{reference length})$$

| Reference | OCR output | Edits | N | CER before | CER after `fix_ocr` |
|---|---|---|---|---|---|
| `Invoice No. 4417` | `Invoice N0. 4417` | 1 | 16 | 1/16 = **0.062** | **0.000** |
| `Total: PKR 12,500` | `Total: PKR 12,5O0` | 1 | 17 | 1/17 = **0.059** | **0.000** |
| `Due 30/09/2026` | `Due 3O/09/2O26` | 2 | 14 | 2/14 = **0.143** | **0.000** |

The repair rule changes a letter O to 0 only next to digits (and `N0.` to `No.`), so "Total" and "Due" are untouched.

**Synthetic invoice (EasyOCR):**

| Measure | Raw OCR | After correction |
|---|---|---|
| CER | **0.000** | **0.000** |
| Field-level accuracy (mobile, amount, date, invoice no.) | 4/4 = 1.00 | 4/4 = 1.00 |

The raw CER is already zero because the image is clean rendered text (one font, black on white, horizontal, sharp, no noise or skew), which is far easier than a phone photo of a real report. A CER of 0.059 can still mean a wrong amount, since one wrong character breaks the whole field, which is why field-level accuracy is reported as well. I expect a higher raw CER on my own scans and will measure it on five real scans with hand-typed reference text.

---

### 6. Pipeline: image model + text evidence -> one masked JSON record

Late fusion with transparent rules (no learned weight, because there are no labelled image-note pairs):

$$\text{REFER}\iff\big(p_{\text{tumor}}\ge0.986\big)\;\text{OR}\;\big(\text{urgent terms}\neq\varnothing\big),\qquad \text{general form: } p=w\,p_{\text{image}}+(1-w)\,p_{\text{text}}$$

| Component | Setting |
|---|---|
| Image model | Week 6 ResNet-18, classes `['no_tumor', 'tumor']`, 224 px, grayscale |
| Operating threshold | 0.986 (Week 6 validation point); logit gap $\ln\frac{0.986}{0.014}=4.25$ |
| Text evidence | spaCy + domain entity ruler, regex fields, urgent terms (headache, seizure, vomit, ...) |
| Masking | `mask_pii` runs **first**: CNIC, mobile, e-mail and MRN replaced by `[CNIC]`, `[PHONE]`, `[EMAIL]`, `[MRN]` |

**Example record** (Te-gl_1.jpg from the dataset's Testing folder; entities shown compactly):

```json
{
  "image": "Te-gl_1.jpg",
  "image_probabilities": {"no_tumor": 0.004, "tumor": 0.996},
  "note_masked": "Patient Imran Qureshi, aged 46, has had headaches for 3 weeks with vomiting and one seizure on 10/11/2026. MRI shows a 2.5 cm lesion on T2. Referred from Mayo Hospital Lahore. [MRN]. CNIC [CNIC], family mobile [PHONE].",
  "text_evidence": {
    "entities": [["Imran Qureshi","PERSON"], ["46","DATE"], ["3 weeks","DATE"], ["10/11/2026","DATE"],
                 ["2.5 cm","QUANTITY"], ["T2","CARDINAL"], ["Mayo Hospital","ORG"], ["Lahore","GPE"],
                 ["CNIC","ORG"], ["CNIC","ORG"]],
    "fields": {"date": ["10/11/2026"], "tumour_size": ["2.5 cm"], "age": ["46"], "sequence": ["T2"]},
    "urgent_terms": ["headache", "seizure", "vomit"]
  },
  "decision": "REFER",
  "reasons": ["image model: P(tumor) = 1.00", "note mentions: headache, seizure, vomit"],
  "flags": [],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

(The reason text shows `1.00` only because 0.996 is printed to two decimals.)

**Reasons check:** $0.996\ge0.986$ so the image reason fires; the note contains three urgent terms so the text reason fires; both are positive, so no disagreement flag.

**Wrong entities in `text_evidence`.** Against my five-entity gold list for the note, 5 of 10 entities are correct and none is missed: $P=5/10=0.50$, $R=5/5=1.00$, $F_1=0.667$. The wrong or unexpected ones are `46` (an age, labelled DATE), `T2` (an MRI sequence, labelled CARDINAL) and two `CNIC` entries labelled ORG (the word "CNIC" and the `[CNIC]` placeholder). They do not change the decision, but the two ORG entries would be harmful as stored data, so placeholders should be filtered out of the entity list.

**Image and note disagree:**

| Case | $p_{\text{tumor}}$ | Urgent terms | Decision | Flag |
|---|---|---|---|---|
| A: healthy image (Te-no_1.jpg) + urgent note | 0.017 (< 0.986) | headache, seizure, vomit | REFER | IMAGE/NOTE DISAGREE |
| B: tumour image (Te-gl_1.jpg) + routine note | 0.996 (>= 0.986) | none | REFER | IMAGE/NOTE DISAGREE |

A missed tumour costs more than an extra review, so any positive source keeps the case on REFER and the flag sends it to a human reviewer. The disagreement itself is the useful signal.

---

### 7. Privacy

- **Masked before extraction.** `case_record` calls `mask_pii(note)` first and runs entity and field extraction on the masked text. Extraction copies text, so extracting from the original note would put the identifier back into `text_evidence` (the bug described in the lab).
- **Leak check.** The output JSON was searched for `35202-1234567-1`, `0300-1234567`, `35202`, `1234567` and `4471823`: **no fragments found**.
- **Scope.** The note still contains a patient name (names are not masked by the lab's rules), so the record is for internal decision support, not for sharing.
- All names, numbers and dates in the examples are invented.

---

### 8. Check Your Understanding

| # | Question | Working | Answer |
|---|---|---|---|
| 1 | Smooth IDF, word in 9 of 1,000 documents | $\ln\frac{1001}{10}+1=\ln(100.1)+1=4.6062+1$ | **5.606** |
| 2 | Why can lowercasing hurt a spam classifier? | "FREE", "WIN", "CLAIM" are shouted; lowercasing merges them with ordinary "free", "win", so the all-caps cue is lost. (scikit-learn's `TfidfVectorizer` lowercases by default, so my models do this too.) | see working |
| 3 | 10 gold, 8 predicted, 6 exactly right | $P=\frac68=0.750$, $R=\frac6{10}=0.600$, $F_1=\frac{2(0.75)(0.60)}{1.35}$ | **P 0.750, R 0.600, F1 0.667** |
| 4 | CER of "PKR 4,5OO" against "PKR 4,500" (9 characters) | 2 substitutions: $\frac{2+0+0}{9}$ | **0.222** |
| 5 | Image 0.30, text 0.80, $w=0.5$ | $0.5(0.30)+0.5(0.80)=0.15+0.40$ | **0.55** |
| 6 | Why mask identifiers before extraction? | extraction copies text, so unmasked identifiers reappear in `text_evidence`, logs and shared records; masking first means no later step ever sees them | see working |

---

### Limitations

- The held-out NER set has only 18 entities and shares formats with the rule-writing set; a score of 1.000 is a small-sample result.
- spaCy already handled every organisation in the held-out sentences, so my hospital-suffix pattern and city list are not yet tested.
- The OCR test used one clean synthetic image; real scans are still to be measured.
- The 0.986 operating threshold comes from my Week 6 validation set and is not stored in the checkpoint; the image reason will not fire for a tumour image scoring, say, 0.90.
- This is decision support only; a qualified human decides.
