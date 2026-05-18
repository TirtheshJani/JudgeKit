# JudgeKit tech note (paper/)

Build:

```bash
cd paper
latexmk -pdf tech_note.tex
```

Numbers (Fleiss kappa, Krippendorff alpha, per-vendor spend, pairwise kappa
heatmap) are filled from the headline run report. After running
`judgekit report headline_v01 --out paper/figures/report.csv`, regenerate the
heatmap from the `pairwise_kappa` rows.
