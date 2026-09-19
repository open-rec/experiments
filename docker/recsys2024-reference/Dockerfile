FROM openrec/recsys2024-reference-core:local

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

# Extend the isolated tree-model layer with the published neural-model stack.
RUN /opt/conda/bin/python -m pip install --no-cache-dir \
    --index-url "${PIP_INDEX_URL}" \
    hydra-slayer==0.5.0 \
    universal-pathlib==0.2.2 \
    optuna==3.6.1 \
    lightning==2.2.2 \
    pytorch-lightning==2.2.2 \
    transformers==4.39.3 \
    accelerate==0.29.1 \
    tensorboard==2.16.2 \
    timm==1.0.3 \
    pytorch-toolbelt==0.6.3 \
    opencv-python==4.10.0.84 \
    tokenizers==0.15.2 \
    nltk==3.8.1 \
    pandarallel==1.6.5 \
    loguru==0.7.2 \
    typer==0.12.2 \
    pydantic==2.7.0

ENV PATH=/opt/workspace/.local/bin:/opt/conda/bin:${PATH}
ENV PYTHONPATH=/workspace/reference/kfujikawa/src:/workspace/reference/kami
WORKDIR /workspace/reference

CMD ["bash"]
