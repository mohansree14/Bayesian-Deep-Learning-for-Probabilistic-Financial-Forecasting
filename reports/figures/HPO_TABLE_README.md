# HPO Search Space Table Image

## Generated Table

**Files:**
- **PNG:** `hpo_search_space_table.png` (327 KB, 300 DPI)
- **PDF:** `hpo_search_space_table.pdf` (43 KB, Vector)

## Table Contents

The image contains a professional table showing:

| Hyperparameter | Type | Min Value | Max Value | Step Size | Possible Values | Distribution | Configs |
|---|---|---|---|---|---|---|---|
| **Hidden Size** | Integer | 64 | 256 | 64 | 64, 128, 192, 256 | Uniform | 4 |
| **Num Layers** | Integer | 1 | 3 | 1 | 1, 2, 3 | Uniform | 3 |
| **Dropout** | Float | 0.0 | 0.4 | Continuous | 0.0–0.4 | Uniform | ∞ |
| **Learning Rate** | Float | 0.0001 | 0.005 | Continuous | 1e-4 to 5e-3 | Log-Uniform | ∞ |
| **Batch Size** | Categorical | — | — | — | 32, 64, 128 | Discrete | 3 |

## Features

✅ **Professional Design:**
- Dark blue header with white text
- Alternating row colors (gray/white) for readability
- Clear borders and cell spacing
- Bold hyperparameter names

✅ **Comprehensive Information:**
- All 5 hyperparameters detailed
- Type, range, and distribution information
- Possible values explicitly listed
- Configuration count per parameter

✅ **Additional Details:**
- Optimization framework (Optuna TPE)
- Objective function (Minimize MSE)
- Number of trials (10)
- Model type (LSTM Regressor)
- Total search space size

✅ **High Quality:**
- 300 DPI PNG for presentations and documents
- Vector PDF for publications and scaling
- Professional color scheme
- Clear typography

## Usage

### In Reports
- Insert PNG directly into Word/PowerPoint
- Use PDF for LaTeX documents
- Professional quality for publications

### In Presentations
- High-resolution for projection
- Clear and readable from distance
- Professional appearance

### Regenerate
```bash
python generate_hpo_table.py
```

## Customization

To modify the table, edit `generate_hpo_table.py`:
- Change colors in the color definitions
- Adjust cell dimensions
- Modify content in the `data` list
- Update footer information

---

**Generated:** October 14, 2025  
**Script:** `generate_hpo_table.py`  
**Purpose:** Visual representation of HPO search space for reports
