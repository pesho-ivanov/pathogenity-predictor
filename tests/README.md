# Tests

Unit tests live here, outside `notebooks/src/`. Run them from the repository root:

Fresh CPU setup and tests (without installing CUDA or downloading datasets):

```sh
python3.12 scripts/setup.py --profile cpu --test
```

This omits `test_q9`, which needs PyTorch. Q11 input/reporting tests run on CPU;
its optimizer and checkpoint tests skip when PyTorch is unavailable. For the full suite, use the GPU
environment described in the [README setup section](../README.md#setup), then run:

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

Comparison tests cover completed-notebook result preservation, cohort and output
integrity, rejection of partial local exports, and stable notebook links when
diagnostic notebooks share a question number.

Q12 tests cover magnitude retention and finite gradients, training-only scaling,
selection/patience rules, validation time reserves, matched batch cursors,
optimizer/checkpoint resume equivalence, and rejection of sampled benchmark exports.

The source package retains compatibility with existing `src.test_*` and
`notebooks.src.test_*` imports used by completed notebook workflows. This keeps
frozen data and running model source hashes unchanged after moving the files.
