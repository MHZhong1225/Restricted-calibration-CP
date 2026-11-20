# SGCP project page

Static research website for **Stochastic Grouping Conformal Prediction for Effective Subgroup Reliability**. No build tools or external runtime dependencies are required.

## Preview

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

Open http://localhost:8000.

## Publish

GitHub repository **Settings → Pages → Deploy from a branch**, select `main` and `/docs`.

Project URL: https://mhzhong1225.github.io/Restricted-calibration-CP/

## Update

- `index.html`: narrative, authors, affiliations, manuscript citation, resource links.
- `app.js`: benchmark values transcribed from manuscript Table 1. Metrics are `[mean, standard deviation]`; an unreported method is `null`.
- `style.css`: responsive layout and appearance.
- `assets/sgcp-paper.pdf`: the supplied `Nips2026_2_Zhong.pdf` manuscript.
- The five SVG figures are vector exports of the supplied method and benchmark PDFs.
