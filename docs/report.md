# Final report

The final academic deliverable for this project is the written course report:

**Modeling Iowa Gambling Task Behavior: A Comparison of Q-learning and PVL-Delta**

The report is maintained as a LaTeX project under the repository-level `report/` directory. The compiled PDF is committed alongside its source so the completed deliverable can be read directly without rebuilding it.

## Repository location

```text
report/
├── report.tex          # Main LaTeX entry point
├── preamble.tex        # Shared packages and formatting
├── references.bib      # Bibliography database
├── sections/           # Main sections and appendices
├── figures/            # Report-ready figures
├── tables/             # Report-ready LaTeX tables
└── report.pdf          # Compiled final-project deliverable
```

- [Open the compiled report on GitHub](https://github.com/orfadida2000/igt-model-comparison/blob/main/report/report.pdf)
- [Browse the report source on GitHub](https://github.com/orfadida2000/igt-model-comparison/tree/main/report)

## Analysis represented in the report

The report presents the final corrected 617-participant analysis. It uses the corrected PVL-Delta result set and the downstream final analysis summarized in [Final results](final-results.md).

The final analysis includes:

- participant-level Q-learning and PVL-Delta fits;
- AIC and BIC model comparison;
- the targeted PVL-Delta optimization correction;
- Q-learning inverse-temperature sensitivity and boundary diagnostics;
- BCa bootstrap confidence intervals;
- paired Wilcoxon signed-rank tests;
- exact binomial confidence intervals and tests for model win rates;
- parameter, convergence, boundary, uniform-choice, and source-study diagnostics.

The Python analysis pipeline produces the numerical, tabular, graphical, and text artifacts that support the written report. It does **not** generate the prose or LaTeX document automatically.

## Build the PDF

A TeX distribution providing `latexmk`, `pdflatex`, and BibTeX is required.

From the repository root:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

The compiled output is:

```text
report/report.pdf
```

The report-specific figures and tables are already stored under `report/`, so rebuilding the checked-in PDF does not require rerunning the Python analysis first.

## Reproducibility scope

Rebuilding `report.pdf` and regenerating the scientific results are separate operations:

1. **Document build** reproduces the PDF from the checked-in LaTeX source, figures, tables, and bibliography.
2. **Scientific regeneration** reruns model fitting, the targeted correction workflow, and final result analysis from the project data.

See [Reproducibility](reproducibility.md) for the full distinction and workflow links.
