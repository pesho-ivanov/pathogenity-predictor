# Tests

Unit tests live here, outside `notebooks/src/`. Run them from the repository root:

Fresh CPU setup and tests (without installing CUDA or downloading datasets):

```sh
python3.12 scripts/setup.py --profile cpu --test
```

This omits `test_q9`, which needs PyTorch. For the full suite, use the GPU
environment described in [SETUP.md](../SETUP.md), then run:

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

The source package retains compatibility with existing `src.test_*` and
`notebooks.src.test_*` imports used by completed notebook workflows. This keeps
frozen data and running model source hashes unchanged after moving the files.
