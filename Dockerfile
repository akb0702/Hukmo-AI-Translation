FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq git build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY bake_model.py /tmp/bake_model.py
RUN --mount=type=secret,id=hf_token \
    HUGGING_FACE_HUB_TOKEN=$(cat /run/secrets/hf_token) \
    python3 /tmp/bake_model.py

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update -qq && \
    apt-get install -y -qq git build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY --from=builder /model /model
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

COPY app/ /app/
WORKDIR /app

ENV MODEL_DIR=/model \
    TRANSLATE_BATCH_SIZE=8 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8080

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "::", "--port", "8080", "--workers", "1"]
