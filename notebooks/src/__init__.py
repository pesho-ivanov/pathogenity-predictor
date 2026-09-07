"""Implementation for the research notebooks."""

from pathlib import Path

# Existing notebook test runners use src.test_* (or notebooks.src.test_*).
# Keep those imports working after moving the files, without changing the
# source hashes pinned by completed datasets and the active model evaluation.
__path__.append(str(Path(__file__).resolve().parents[2] / 'tests'))
