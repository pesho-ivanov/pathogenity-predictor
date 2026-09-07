# Set up on another machine

Use **Linux x86-64 and Python 3.12**. The CPU profile supports data preparation,
published-score evaluation and notebook analysis. The GPU profile adds the
compiled Evo2 stack used by the recorded experiments.

## Clone and install

On Ubuntu 24.04, install the small system prerequisites first:

```bash
sudo apt-get update
sudo apt-get install -y git curl python3.12-venv
git clone --recurse-submodules https://github.com/pesho-ivanov/pathogenity-predictor.git
cd pathogenity-predictor
python3.12 scripts/setup.py --profile cpu --test
.venv/bin/jupyter lab
```

Select the **Gamow** kernel. Setup installs it inside the virtual environment;
start Jupyter from that same environment. Run the notebooks in `notebooks/`,
where their `from src import ...` imports resolve correctly.

The script reads the root `requirements.txt`, omits only `vtx` and the editable
Evo2 package for CPU setup, and verifies the pinned package versions and imports.
It does not download datasets or models. CPU tests omit `test_q9`, whose optimizer
tests require PyTorch; the GPU profile runs the entire suite.

Rerun setup to repair an installation, or check it without installing anything:

```bash
python3.12 scripts/setup.py --profile cpu --check
```

`--venv /path/to/environment` selects another environment. CPU environments are
isolated; GPU environments inherit the NVIDIA container's compiled packages.
Setup refuses to change an existing environment to the other profile. Use a
fresh path or a separate checkout when changing profiles.

## NVIDIA GPU machine

Choose **`nvcr.io/nvidia/pytorch:25.04-py3`** as the rental provider's container
image. It supplies the pinned PyTorch/Transformer Engine stack and CUDA 12.9.
See [NVIDIA's release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-04.html).
Inside the container, clone the repository onto persistent storage and run:

```bash
git clone --recurse-submodules https://github.com/pesho-ivanov/pathogenity-predictor.git
cd pathogenity-predictor
python scripts/setup.py --profile gpu --test
.venv/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Keep Jupyter's default token authentication enabled and use the provider's
authenticated proxy or an SSH tunnel. The setup check runs Transformer Engine
and Flash Attention kernels on every visible GPU without downloading a model.
An incompatible driver or missing compiled library fails during setup, before
an experiment starts. Model-specific numerical checks still run in the workflows.

On a host with Docker, an NVIDIA driver and NVIDIA Container Toolkit already
installed, launch the same environment from a fresh repository checkout:

```bash
docker run --rm -it --gpus all --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD:/workspace/gamow" \
  -v gamow-hf-cache:/root/.cache/huggingface \
  -w /workspace/gamow \
  nvcr.io/nvidia/pytorch:25.04-py3 bash
```

Then run the GPU setup command inside that shell. For a remote Docker host,
open the tunnel from your local machine:

```bash
ssh -L 8888:127.0.0.1:8888 user@host
```

Open the Jupyter URL at `localhost:8888` using the token printed by Jupyter.
The repository mount preserves data, notebooks and results; the named volume
preserves Hugging Face downloads. Provider containers need equivalent persistent
storage mounts. Use the provider's NVIDIA image directly when nested Docker is
unavailable.

For planning, allocate around **64 GiB RAM and 30 GiB free disk** for data
preparation, or **128 GiB RAM and 200 GiB free SSD space** for the GPU workflows.
These are practical starting allocations, not measured minimum requirements.
Keeping multiple source and converted checkpoints needs additional disk space.

### BioNeMo environment

The existing BioNeMo bootstrap uses a second environment at `.venv/q9`, installed
from pinned source revisions. To prepare its packages without starting training:

```bash
.venv/bin/python -c 'from notebooks.src.q9_environment import setup; setup()'
```

The first install clones dependencies and compiles CUDA extensions. Compilation
automatically targets the visible GPUs, including Hopper's `9.0` architecture;
an explicit `TORCH_CUDA_ARCH_LIST` overrides detection. The selected architectures
are recorded so a cache built for a different GPU triggers a rebuild.

This preserves the current BioNeMo implementation. **Distributed 7B LoRA is not
implemented yet.** A successful setup check establishes environment readiness,
not a completed training experiment or a measured speedup.

## Prepare data and execute notebooks

Start with **Q0**, then **Q1**. They acquire checksum-pinned ClinVar and GRCh38
inputs automatically; Q1 reconstructs the frozen July partitions without needing
an old machine's result cache. See [data sources and sizes](data/README.md).
Q8 additionally downloads published predictor scores. PrimateAI-3D requires
licensed data and remains explicitly unavailable until those data are supplied.

For unattended execution, these commands start fresh kernels and save notebooks
with execution counts and outputs retained. Run them from the repository root:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=gamow --ExecutePreprocessor.timeout=-1 \
  notebooks/Q0-clinvar-summary.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=gamow --ExecutePreprocessor.timeout=-1 \
  notebooks/Q1-clinvar-split.ipynb
```

Only continue after successful execution. Use the same command with another
notebook path when its inputs are ready. Refresh the README comparison after
completing result-producing notebooks:

```bash
.venv/bin/python -c 'from notebooks.src.comparison import refresh; refresh()'
```

The current full cohort is **46,888 training and 17,927 validation variants**.
Q2/Q9/Q10 include historical pilot experiments; Q9 explicitly requires the old
pilot hashes. Updating those experiments to the full July cohort is separate
work. Keep their guards intact: a dataset
or protocol mismatch is a reproducibility error, not an installation error.

## Move existing work

Git does not include `data/`, generated `notebooks/results/` artifacts, model
caches or virtual environments. Public inputs can be downloaded again. To retain
expensive downloads, frozen protocols and resumable checkpoints, copy the data
and results into the corresponding directories of the destination checkout:

```bash
rsync -aP data/ user@host:/workspace/gamow/data/
rsync -aP notebooks/results/ user@host:/workspace/gamow/notebooks/results/
```

Create the destination directories first and substitute your actual paths. Copy
the Hugging Face cache separately if you want to reuse it; its default location
is `~/.cache/huggingface`. Recreate `.venv` and `.venv/q9` on the destination:
virtual environments contain absolute paths and CUDA extensions may depend on
the source GPU architecture.

Use the commit or archived implementation recorded with an experiment to resume
it. Source hashes, inputs, labels and preprocessing must match its protocol.
Preserve old result directories before starting an experiment with changed
sources; do not delete or rewrite the protocol to bypass verification. Copied
result caches do not replace executing a notebook before delivering it.

## Troubleshooting

- **Externally managed Python:** use `scripts/setup.py`; it installs into a venv.
- **Wrong package versions:** use the documented NVIDIA image for GPU work and
  rerun setup. The script protects PyTorch, Transformer Engine and Flash Attention.
- **Missing Evo2:** GPU setup initializes the pinned Git submodule automatically.
  Local source edits or another submodule revision require review before updating.
- **CUDA architecture error:** recreate the compiled environment on the new GPU.
  BioNeMo detects its architecture; a saved `TORCH_CUDA_ARCH_LIST` can override it.
- **No `src` module / wrong kernel:** launch `.venv/bin/jupyter lab`, select Gamow,
  and execute the notebook from `notebooks/`.
- **Stale artifact or cohort:** use matching archived sources and inputs, or
  prepare a new experiment in a fresh checkout. Keep the checksum checks enabled.
