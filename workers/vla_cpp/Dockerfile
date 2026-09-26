FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    cppzmq-dev \
    git \
    libprotobuf-dev \
    libzmq3-dev \
    pkg-config \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Keep the established compiler layer cached when adding headless simulation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglib2.0-0 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY policykit ./policykit
COPY configs ./configs
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "uv" "huggingface_hub>=0.30,<1.0" '.[quantize]'

COPY . .
ENTRYPOINT ["policykit"]
