ARG ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:6.1.0
FROM ${ISAAC_IMAGE}
ARG ISAAC_UID=1234

# Build tools run as root; simulation retains NVIDIA's runtime UID.
USER root
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/sim-worker/requirements.txt
RUN /isaac-sim/python.sh --no-ros-env -m pip install --no-cache-dir \
    -r /opt/sim-worker/requirements.txt

COPY --chown=${ISAAC_UID}:${ISAAC_UID} sim_worker /opt/sim-worker/sim_worker
COPY --chown=${ISAAC_UID}:${ISAAC_UID} scenes /opt/sim-worker/scenes
COPY --chown=${ISAAC_UID}:${ISAAC_UID} tests /opt/sim-worker/tests
COPY --chown=${ISAAC_UID}:${ISAAC_UID} build.sh /opt/sim-worker/build.sh
COPY --chown=${ISAAC_UID}:${ISAAC_UID} demo.yaml /opt/sim-worker/demo.yaml

ARG ISAAC_IMAGE
LABEL simulation.isaac.base="${ISAAC_IMAGE}"
ENV PYTHONUNBUFFERED=1
WORKDIR /opt/sim-worker
USER ${ISAAC_UID}:${ISAAC_UID}
ENTRYPOINT ["/isaac-sim/python.sh", "--no-ros-env", "-m", "sim_worker"]
CMD ["--manifest", "/opt/sim-worker/demo.yaml", "--output-dir", "/tmp/sim-results"]
