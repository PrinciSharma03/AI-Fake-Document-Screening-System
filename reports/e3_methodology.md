# E3 methodology: synthetic scans and edit traces

Files: src/data_processing/e3_generate.py, e3_features.py, src/evaluation/run_e3.py
Data (not on GitHub): data/raw/e3/ (images, labels), data/processed/e3/ (features). Original dataset (E0-E2) unchanged.

## 1. Clean page
A fictional marksheet is drawn by draw_marksheet() (1000 x 1300 px; name, roll number, 5 subjects with marks and grades, total, percentage, result). All names and marks are invented.

## 2. Scan / photo simulation (same recipe for genuine and forged)
Per document, parameters are drawn from these ranges (uniform): rotation +-0.3 degrees; Gaussian blur sigma 0.4 to 1.2 px; Gaussian noise sigma 2 to 8 grey levels; contrast factor 0.92 to 1.08; brightness shift -10 to +10; final JPEG quality 60 to 90.
A forged copy starts from the SAME scanned page as its genuine twin and is saved with the SAME JPEG quality, so both classes share format, size and compression recipe.

## 3. Forgery styles (one field is changed: name, roll number, one subject mark, or total)
| Style | Used in | How the edit is made | Trace it leaves |
|---|---|---|---|
| 1 retype | training | Cover the field with the local paper colour, type the new value in the same font and size, blur it by sigma 0.3 to 1.0, add noise at 20% to 50% of the page noise | Patch is smoother and less noisy than the scan around it |
| 2 paste | training | Copy the same field from a different document's scan, shift it by up to 2 px, blend the border over 4 px | Patch carries another scan's noise, brightness and blur |
| 3 resample | UNSEEN test | Draw the new value at 2x size, shrink with area interpolation, JPEG-compress the patch alone at quality 35 to 55 | Resampling and local compression artifacts |

## 4. Splits
- Train: 60 groups (60 genuine + 60 forged; styles 1 and 2 alternate). Evaluated with grouped 5-fold cross-validation.
- Unseen test: 40 NEW groups (40 genuine + 40 forged, style 3 only). Group ids never appear in training.
- An original and its forged copy share one group_id and are never separated.

## 5. Features
- Baseline: the 5 agreed features from extract_features (blur_score, noise_level, edge_density, contrast, ocr_confidence).
- Regions: 10 regions (name, roll number, 5 subject marks, total, percentage, result) x 6 measures (paper brightness, text darkness, paper noise, edge share, sharpness, compression-grid strength) = 60.
- Relative: each region measure minus the median of that measure over the 10 regions of the same page = 60.

## 6. Evaluation
Logistic Regression, Random Forest, SVM; accuracy, precision, recall, F1, confusion matrix; results in reports/results_table.csv. Test set used once per feature set; generator and features not tuned on test results.

## 7. Limitations
- Everything is synthetic; results show detection of the simulated traces, not of real forgeries.
- Region boxes follow our own template and need a fixed layout (rotation is kept very small for this reason). Other layouts would need alignment or detection of the fields.
- Style 3 shares the idea "the patch was made separately" with styles 1 and 2, so it is unseen but not unrelated.
- 80 unseen test rows give about +-11 points of statistical uncertainty.