# Project Documentation

This directory contains comprehensive documentation for the Financial Time Series Forecasting project.

## 📚 Main Documentation

### Complete Reports

1. **[PROJECT_ANALYSIS_REPORT.md](PROJECT_ANALYSIS_REPORT.md)**
   - Executive summary
   - System architecture
   - Technology breakdown
   - Model details (LSTM, MC Dropout, Bayesian, Transformer)
   - Data pipeline and preprocessing
   - Training process and evaluation metrics
   - Backtesting system
   - Application features
   - Deployment approach
   - Results comparison
   - Future improvements

2. **[FIGURES_DOCUMENTATION.md](FIGURES_DOCUMENTATION.md)**
   - Figure 6: AAPL Price History (2015-2024)
   - Figure 7: Feature Correlation Heatmap
   - Figure 8: Training/Validation Loss Curves
   - Figure 9: Uncertainty Bands Visualization
   - Figure 10: Cumulative Returns Comparison
   - Complete technical specifications
   - Usage guidelines
   - LaTeX and Markdown examples

3. **[HPO_SEARCH_SPACE.md](HPO_SEARCH_SPACE.md)**
   - Hyperparameter optimization configuration
   - Search space definitions
   - Parameter rationale and trade-offs
   - Optuna TPE algorithm details
   - Example configurations
   - Interpretation guidelines

### Quick References

4. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**
   - At-a-glance project overview
   - Quick start commands
   - Common tasks
   - File structure
   - Key metrics

5. **[HPO_QUICK_REFERENCE.md](HPO_QUICK_REFERENCE.md)**
   - HPO search space summary
   - Parameter ranges
   - Usage examples
   - Best practices

## 📊 Tables and References

### Tables Directory (`tables/`)

- **[TECHNICAL_INDICATORS_TABLE.md](tables/TECHNICAL_INDICATORS_TABLE.md)**
  - All 21 technical indicators
  - Detailed formulas
  - Parameters and purposes
  - Interpretation guidelines
  - Category breakdowns

- **[HPO_SEARCH_SPACE_TABLE.md](tables/HPO_SEARCH_SPACE_TABLE.md)**
  - 12 comprehensive reference tables
  - Parameter details
  - Trade-offs analysis
  - Optimization settings

- **[HPO_TABLE_SINGLE.md](tables/HPO_TABLE_SINGLE.md)**
  - Single-page HPO reference
  - Report-ready format

### Figures Directory (`figures/`)

All generated visualizations and table images:

**Documentation Figures:**
- `figure_6_aapl_price_2015_2024.png` / `.pdf`
- `figure_7_correlation_heatmap.png` / `.pdf`
- `figure_8_training_validation_loss.png` / `.pdf`
- `figure_9_uncertainty_bands.png` / `.pdf`
- `figure_10_cumulative_returns.png` / `.pdf`

**Table Images:**
- `hpo_search_space_table.png` / `.pdf`
- `technical_indicators_table.png` / `.pdf`

**Figure Documentation:**
- `FIGURE_9_README.md` - Uncertainty Bands guide
- `FIGURE_10_README.md` - Cumulative Returns guide
- `HPO_TABLE_README.md` - HPO table usage

## 🗂️ Organization

```
reports/
├── README.md                         # This file
├── PROJECT_ANALYSIS_REPORT.md        # Complete project analysis
├── QUICK_REFERENCE.md                # Quick reference guide
├── FIGURES_DOCUMENTATION.md          # All figures documented
├── HPO_SEARCH_SPACE.md              # HPO detailed documentation
├── HPO_QUICK_REFERENCE.md           # HPO quick guide
├── final_report.md                  # Final report template
├── intern_plan.md                   # Internship plan
├── slides.md                        # Presentation slides
├── figures/                         # Generated visualizations
│   ├── figure_*.png/pdf            # Documentation figures
│   ├── *_table.png/pdf             # Table images
│   └── *_README.md                 # Figure guides
└── tables/                          # Reference tables
    ├── TECHNICAL_INDICATORS_TABLE.md
    ├── HPO_SEARCH_SPACE_TABLE.md
    └── HPO_TABLE_SINGLE.md
```

## 📖 Documentation Types

### For Developers
- `PROJECT_ANALYSIS_REPORT.md` - Technical deep dive
- `FIGURES_DOCUMENTATION.md` - Visualization details
- `tables/TECHNICAL_INDICATORS_TABLE.md` - Indicator formulas

### For Users
- `QUICK_REFERENCE.md` - Getting started quickly
- `HPO_QUICK_REFERENCE.md` - HPO usage
- `figures/*_README.md` - Figure guides

### For Reports/Publications
- All PNG/PDF figures (300 DPI, publication quality)
- Table images (clean, professional)
- `tables/HPO_TABLE_SINGLE.md` - Single-page tables

## 🎯 Usage

### For Academic Papers
```latex
\includegraphics[width=\textwidth]{figures/figure_9_uncertainty_bands.pdf}
```

### For Presentations
- Use high-resolution PNG files from `figures/`
- Reference table images for clear data presentation

### For README/Markdown
```markdown
![HPO Search Space](reports/figures/hpo_search_space_table.png)
```

## 🔄 Updating Documentation

### Regenerate Figures
```bash
python utils/generate_figures.py
```

### Regenerate Tables
```bash
python utils/generate_hpo_table.py
python utils/generate_indicators_table.py
```

## 📝 Documentation Standards

All documentation follows:
- **Comprehensive**: Detailed technical reports
- **Accessible**: Quick reference guides
- **Visual**: High-quality figures and tables
- **Professional**: Publication-ready quality

## 🆘 Need Help?

- **Quick answers**: Check `QUICK_REFERENCE.md`
- **Detailed info**: See `PROJECT_ANALYSIS_REPORT.md`
- **Specific topics**: Use table of contents in each document

---

**Last Updated**: October 14, 2025  
**Maintained by**: ML Project Team  
**Purpose**: Comprehensive project documentation for research, development, and publication
