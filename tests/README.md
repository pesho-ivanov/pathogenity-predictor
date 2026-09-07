# Tests

Unit tests live here, outside `notebooks/src/`. Run them from the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

The source package retains compatibility with existing `src.test_*` and
`notebooks.src.test_*` imports used by completed notebook workflows. This keeps
frozen data and running model source hashes unchanged after moving the files.
