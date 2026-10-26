# LaTeX report

`main.pdf` is the compiled two-page report, including references. `main.tex` is the main source. The project is self-contained: upload this directory's contents to Overleaf and select `main.tex`.

For local compilation run `pdflatex main.tex` twice, or `tectonic main.tex` once. The finished PDF was compiled with Tectonic 0.17.0 and visually checked at A4 size. Standard packages only; no institutional template or paid font is required.

`generated_metrics.tex` and the two figures are generated from actual experiment outputs by `python -m privacyguard.report_assets` from the repository root. Do not manually edit experimental numbers.

Before course submission, replace the explicitly labelled account-profile/publication-pending line in `team.tex` with the verified project repository URL, then recompile. The supplied profile is https://github.com/JaimilNPatel; the project repository has not been published by this package.
