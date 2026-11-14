# Digit Recognition Robustness Under Random Patch Noise

An experimental investigation of whether training a small neural network with localized pixel corruption improves robustness to similarly corrupted handwritten digits.

> **Key result (original saved run):** Averaged over five initialization seeds, 2 × 2 patch-noise augmentation improved accuracy on corrupted test digits from **90.7% to 92.9%** (+2.2 percentage points), while clean test accuracy changed from 96.8% to 96.6%. These results are specific to the experiment below, not a claim of general statistical significance.

## Research question

Can an MLP trained with random local pixel noise become more accurate on images corrupted in the same way, without substantially sacrificing clean-image accuracy?

**Important distinction:** The experiment does *not* erase pixels to zero. A randomly positioned square patch is replaced with independent **Uniform(0, 1)** pixel values. We therefore call this **random patch-noise augmentation**, not strict patch erasure or simulated physical occlusion.

## Method

- **Dataset:** scikit-learn's `load_digits` (1,797 grayscale 8 × 8 handwritten digits, normalized to [0, 1]).
- **Split:** stratified 60% train / 20% validation / 20% held-out test, with fixed splitting seeds.
- **Model:** `MLPClassifier`, one 32-unit ReLU hidden layer, Adam optimizer, batch size 64.
- **Baseline:** train on clean images.
- **Augmented model:** for each training epoch, independently apply random 2 × 2 patch noise to each image with probability 0.5.
- **Training comparison:** 200 `partial_fit` epochs per model; matched initialization seeds; five seeds (0–4).
- **Selection:** compare training patch sizes 2 × 2, 4 × 4, and 6 × 6 on validation data. Then evaluate the selected 2 × 2 configuration on test data.
- **Evaluation:** use the *same fixed corrupted evaluation images* for paired comparisons between clean-trained and augmented models.

## Results

The notebook's original saved-run test results, averaged over the five seeds:

| Training regime | Clean test | Test with 2 × 2 patch noise |
| --- | ---: | ---: |
| Clean-only baseline | 96.8% | 90.7% |
| Noise-augmented (50% probability) | 96.6% | 92.9% |
| **Difference** | **−0.2 pp** | **+2.2 pp** |

All five tested initialization seeds had positive gains on corrupted test images: **[3.33, 2.78, 1.67, 1.11, 2.50] percentage points**. This is evidence for this particular experimental setup; five seeds do **not** amount to a statistical significance test.

## Run the experiment

1. Clone this repository and open its directory.
2. Create a Python virtual environment (recommended).
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Open `notebooks/robustness_experiments.ipynb` with Jupyter or Google Colab and select **Run all**.

The notebook is self-contained and generates the data from scikit-learn; no external dataset download is needed. The random seeds used for splitting, model initialization, and corruption are visible in the code. Library-version differences may lead to slightly different numbers; exact environment versions from the original run were not recorded.

## Repository structure

```text
.
├── notebooks/
│   └── robustness_experiments.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations and next steps

- Low-resolution 8 × 8 digits and one MLP architecture limit generalizability.
- The held-out test set is small and uses one fixed corrupted sample per test image.
- The method tests *patch noise*, not zero masks, real-world occlusions, Gaussian blur, or adversarial robustness.
- Averaging five initialization seeds does not establish statistical significance, and the evaluation is not independently replicated on multiple datasets.
- Next steps: evaluate additional corruption types, report uncertainty intervals, and compare alternative architectures on more realistic datasets.

## Project status

**Learning/research project.** The cleaned notebook preserves the original experiment's methods and saved results; end-to-end reproducibility on a fresh environment should still be independently verified before citing the numbers externally.
