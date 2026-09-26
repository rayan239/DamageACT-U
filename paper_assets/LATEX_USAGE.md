# LaTeX usage

All `.tex` tables in this bundle have been post-processed to avoid raw Unicode
math symbols that commonly fail under `pdflatex`.

Recommended usage:

```latex
\input{paper_assets/tables/main/table03_clean_test_results.tex}
```

The repository tracks editable SVG figures. The generator can also render PNG/PDF copies. For Overleaf, use generated PDF copies when the target venue prefers PDF:

```latex
\includegraphics[width=\linewidth]{paper_assets/figures/main/fig04_clean_gain_vs_post_forest.pdf}
```

The CSV files are the human-auditable numerical sources for each table.
