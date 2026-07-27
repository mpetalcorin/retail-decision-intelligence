# Retail Decision Intelligence

An end-to-end retail data science project integrating demand forecasting, stockout-risk prediction, product-substitution ranking, promotion-effect analysis, explainable machine learning, and inventory optimisation.

![Retail Decision Intelligence](assets/retail_decision_intelligence_infographic_design.png)

## Project scope

The project uses a reproducible synthetic retail panel calibrated from peer-reviewed retail and forecasting literature. It demonstrates how store, product, price, promotion, inventory, and availability signals can be integrated into a unified decision-support workflow.

Core analyses include:

- hierarchical daily demand simulation;
- seasonal-naive and global machine-learning forecasting;
- stockout-censored demand analysis;
- probabilistic forecasting and interval calibration;
- permutation-based feature interpretation;
- stockout-risk classification and calibration;
- product-substitution ranking;
- difference-in-differences promotion analysis;
- inventory-policy optimisation;
- segment-level performance diagnostics.

## Repository structure

```text
assets/       Project infographic and figure montage
manuscript/   Two-column Word manuscript
notebooks/    Executed Jupyter notebook
outputs/      Figures, synthetic data, metrics, and supplementary result tables
src/          Notebook and manuscript generation scripts
```

## Main notebook

Open:

```text
notebooks/Retail_Decision_Intelligence.ipynb
```

The notebook contains the full simulation, statistical analysis, machine-learning workflow, visualisation, interpretation, and literature-grounded assumptions.

## Reproduce the analysis

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook notebooks/Retail_Decision_Intelligence.ipynb
```

## Selected benchmark results

- 31.8% reduction in forecast WAPE relative to the seasonal-naive benchmark.
- Stockout-risk ROC-AUC of 0.747.
- 13.4 percentage-point improvement in top-ranked substitution acceptance.
- 6.9% reduction in normalised inventory decision cost.
- 29.2 percentage-point improvement in service level.
- 93.8% empirical coverage for a nominal 90% prediction interval.

These results describe performance within the synthetic benchmark and should not be interpreted as evidence from a specific retailer or production deployment.

## Manuscript

The journal-style manuscript is available at:

```text
manuscript/Retail_Decision_Intelligence_Full_Manuscript.docx
```

It includes the expanded scientific rationale, methods, results, discussion, embedded figures, peer-reviewed references, and supplementary tables.
