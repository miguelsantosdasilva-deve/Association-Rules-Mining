# Association Rules Mining — Supermarket Dataset

A data mining project applying the **Apriori algorithm** to transactional supermarket data. The workflow covers data preparation in Python, rule extraction in Weka, and interpretation of association metrics with business-oriented analysis.

---

## Overview

The project follows a complete data mining pipeline: reading a raw `.arff` file, cleaning and exporting it as CSV, running the Apriori algorithm in Weka under different parameter configurations, and interpreting the discovered rules in terms of consumer behavior and retail strategy.

---

## Workflow

```
supermarket.arff  →  Python (cleaning)  →  supermarket_clean.csv  →  Weka (Apriori)  →  Rule analysis
```

---

## Data Preparation — Python

The original file is in `.arff` format, native to the Weka ecosystem. The preparation pipeline performs three operations before exporting to CSV:

**1. Load the .arff file**

```python
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np

data, _ = loadarff('supermarket.arff')
df = pd.DataFrame(np.array(data), dtype=str)

print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
```

**2. Remove department columns**

Columns prefixed with `department` are structural metadata with no analytical value for basket mining. They are removed automatically using Pandas string filtering.

```python
dept_cols = df.filter(like='department').columns
df = df.drop(columns=dept_cols)

print(f"Removed {len(dept_cols)} department columns. Remaining: {df.shape[1]}")
```

**3. Remove zero-support columns**

Columns where every value is `?` (absent in all transactions) carry no information and add dimensionality without contributing to any rule. They are identified with a boolean mask and dropped.

```python
empty_cols = df.columns[(df == '?').all()]
df = df.drop(columns=empty_cols)

print(f"Removed {len(empty_cols)} zero-support columns. Final shape: {df.shape}")
assert not (df == '?').all().any(), "Zero-support columns still present after cleaning."
```

**4. Export to CSV**

```python
output_path = 'supermarket_clean.csv'
df.to_csv(output_path, index=False)

print(f"Exported to {output_path} — {df.shape[0]} rows, {df.shape[1]} columns")
```

The `index=False` parameter is required. Without it, Pandas writes its integer row index as an extra column, which Weka would treat as a valid attribute.

---

## Weka Configuration — Apriori

The cleaned CSV was loaded into the Weka Explorer for rule extraction. Two parameter configurations were used.

**Run 1 — Relaxed confidence**

| Parameter | Value |
|---|---|
| minSupport | 0.15 |
| minConfidence | 0.70 |
| numRules | 10 |

Lowering confidence from the default 0.90 diversified the rule set beyond the dominant `bread and cake` patterns, surfacing rules related to produce, dairy, and breakfast items.

**Run 2 — Lift-focused search**

| Parameter | Value |
|---|---|
| minSupport | 0.05 |
| metricType | Lift |
| minMetric | 1.5 |
| numRules | 30 |

Switching the optimization target to Lift shifted focus from raw rule frequency to the strength of association relative to chance, revealing lower-frequency but higher-value patterns.

---

## Rules Selected for Analysis

Three rules were selected to cover distinct consumer scenarios.

| Rule | Antecedent | Consequent | Confidence | Lift | Leverage |
|---|---|---|---|---|---|
| 1 | `vegetables=t` | `fruit=t` | 0.75 | 1.16 | 0.07 |
| 2 | `bread and cake=t` | `milk-cream=t` | 0.70 | 1.10 | 0.05 |
| 3 | `total=high` | `biscuits=t, frozen foods=t, tissues-paper prd=t` | 0.44 | 1.91 | 0.08 |

**Rule 1 — Produce cross-selling**

Customers purchasing vegetables have a 75% probability of also purchasing fruit. The Lift of 1.16 confirms this co-occurrence exceeds what would be expected under independence. The Leverage of 0.07 indicates the combination appears 7 percentage points more frequently than chance would predict. Retail application: co-locate produce sections and display recipe suggestions combining both categories.

**Rule 2 — Breakfast basket**

The `bread and cake → milk-cream` association captures a routine morning or snack purchase pattern. With a Lift of 1.10 and Leverage of 0.05, the association is moderate but consistent. Retail application: promotional bundles combining these items with a shared discount increase average ticket without appearing intrusive.

**Rule 3 — Monthly stock-up**

Transactions with high total value are strongly associated with biscuits, frozen foods, and paper/tissue products (Lift 1.91, the highest in the set). This pattern reflects a full monthly restocking behavior, typically observed at the start of a pay cycle. Retail application: seasonal campaigns with bundled discounts on these categories, timed to early-month spending peaks.

---

## Metrics Reference

| Metric | Formula | Interpretation |
|---|---|---|
| Support | freq(A ∪ B) / N | Proportion of transactions containing both A and B |
| Confidence | P(B \| A) = supp(A ∩ B) / supp(A) | Probability of B given A is present |
| Lift | supp(A ∩ B) / [supp(A) × supp(B)] | Association strength relative to statistical independence; >1 indicates positive association |
| Leverage | supp(A ∩ B) − [supp(A) × supp(B)] | Absolute excess co-occurrence beyond independence; 0 means no association |

Confidence alone can be misleading when the consequent is a high-frequency item. Lift and Leverage correct for this by contextualizing the rule against the baseline probability of each item appearing independently.

---

## Tech Stack

- **Python 3** — data preparation
- `scipy.io.arff` — `.arff` file parsing
- `pandas` / `numpy` — data manipulation and export
- **Weka 3** — Apriori algorithm execution and rule evaluation

---

## Project Structure

```
association_rules_apriori.ipynb       # Data preparation notebook
supermarket.arff                      # Original dataset (Weka format) — not included; see "How to Run"
supermarket_clean.csv                 # Cleaned dataset, generated by the notebook (loaded into Weka)
association_rules_report.pdf          # Full technical report with rule analysis
requirements.txt                      # Python dependencies
LICENSE                               # MIT License
README.md                             # Project documentation
```

---

## How to Run

1. Rename the original file to `supermarket.arff` and place it in the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook association_rules_apriori.ipynb
   ```
4. Load `supermarket_clean.csv` into the Weka Explorer (Classify > Associate > Apriori) and configure parameters as described above.

---

## Notes

Association rules describe correlation, not causation. The patterns identified suggest behavioral tendencies in the observed transaction data and should be validated against current business context before operational use.

---

## License

This project is released under the MIT License — see the [LICENSE](LICENSE) file for details.
