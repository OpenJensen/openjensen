FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY policykit ./policykit
RUN pip install --no-cache-dir '.[quantize]'
ENTRYPOINT ["policykit-worker"]
