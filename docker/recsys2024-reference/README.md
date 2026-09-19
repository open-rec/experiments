# RecSys Challenge 2024 reference environment

This image isolates the winning EB-NeRD solution from all OpenRec services and
runtime images. It uses `openrec/experiments-gpu:local` as an immutable parent
and installs the competition-only Python dependencies in a new image named
`openrec/recsys2024-reference:local`.

The source and dataset mounts are read-only. Only the output mount is writable.
Keep the output on the SSD because the feature pipeline creates large
intermediate files.

```bash
export RECSYS2024_SOURCE=/home/xsank.mz/openrec/tmp/recsys-challenge-2024-1st-place
export EBNERD_DATA_ROOT=/home/xsank.mz/openrec/experiments/data/raw
export RECSYS2024_OUTPUT_ROOT=/ssd/2/xsank.mz/recsys-challenge-2024-reference/output

mkdir -p "$RECSYS2024_OUTPUT_ROOT"
docker build -t openrec/recsys2024-reference-core:local \
  -f docker/recsys2024-reference/Dockerfile.core \
  docker/recsys2024-reference
docker compose -f docker/recsys2024-reference/compose.yaml build
docker compose -f docker/recsys2024-reference/compose.yaml run --rm reference
```

The build uses the Aliyun PyPI mirror by default. Override `PIP_INDEX_URL` only
when a different package index is explicitly required.

The local base currently provides Python 3.11, PyTorch 2.8.0 with CUDA 12.9,
Pandas 2.3.3, PyArrow 23.0.1, scikit-learn 1.7.2, and SciPy 1.17.1. The published
Kfujikawa environment used Python 3.10, PyTorch 2.2.2, Pandas 2.2.1, PyArrow
15.0.2, scikit-learn 1.4.0, and SciPy 1.13.0. Competition-specific libraries
that are sensitive to model behavior follow the published versions. Record the
base-runtime differences when reporting reproduction results.
