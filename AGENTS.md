# Notebook completion

- Save every delivered notebook in its finished, executed state: run all nonempty
  code cells in order from a fresh kernel and retain execution counts and outputs.
- Do not leave result cells uncomputed, clear outputs before saving, or present
  cached result files as a substitute for a fully executed saved notebook.
- Remove unused empty code cells. Never fabricate execution counts or conceal
  errors; resolve failures before calling a notebook finished, and report any
  genuine blocker explicitly.
- Automatic notebook generation and refresh must also execute the notebook before
  saving it. Keep the root comparison synchronized with completed source results.
- Follow the scientific and reproducibility requirements in GOALS.md.
