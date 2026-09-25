FROM openrec/experiments-gpu:local

RUN /opt/conda/bin/python -m pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    "polars>=1.40.1" "optuna>=4.0.0" \
    "xgboost>=3.2.0" "numba>=0.65.1" \
    "faiss-cpu>=1.14.2" "vadersentiment>=3.3.2"

RUN /opt/conda/bin/python -m pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    "matplotlib>=3.10.0"

RUN /opt/conda/bin/python -m pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    "cupy-cuda12x>=13.6.0"

ENV PYTHONPATH=/opt/workspace/.local/lib/python3.11/site-packages
ENV PATH=/opt/workspace/.local/bin:$PATH

WORKDIR /workspace
COPY prepare_clean_split.py /opt/clean/prepare_clean_split.py
