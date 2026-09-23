FROM nvcr.io/nvidia/pytorch:25.05-py3

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# The upstream winner uses three separate environments. This image keeps its
# Python source unchanged while supplying their shared runtime on the local GPU
# base image. The precise source commit is checked by run.py at runtime.
RUN python -m pip install --no-cache-dir \
    "hydra-core==1.3.2" "lightning==2.5.2" "pytorch-lightning==2.5.2" \
    "torchmetrics==1.7.2" "polars==1.30.0" "transformers==4.53.0" \
    "schedulefree==1.4.1" "wandb==0.20.1" "neptune==1.14.0" \
    "loguru==0.7.3" "scikit-learn==1.7.2" "python-dotenv==1.1.1" \
    "rich==14.0.0" "matplotlib==3.10.5" "seaborn==0.13.2"

WORKDIR /output
ENTRYPOINT ["/usr/bin/python", "/opt/reproduce/run.py"]
