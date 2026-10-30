# syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS simulator

ARG DEBIAN_FRONTEND=noninteractive
ARG ASTRA_SIM_REF=4ce9ecfe746a3ee5fe47559a11d91644d641fa7f
ARG BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential cmake libboost-program-options-dev \
    protobuf-compiler libprotobuf-dev python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# The host's old/incomplete submodule and generated binaries are never used.
RUN git init /opt/astra-sim \
    && git -C /opt/astra-sim remote add origin https://github.com/astra-sim/astra-sim.git \
    && git -C /opt/astra-sim fetch --depth 1 origin "${ASTRA_SIM_REF}" \
    && git -C /opt/astra-sim checkout --detach FETCH_HEAD \
    && git -C /opt/astra-sim submodule update --init --recursive --depth 1 \
       extern/graph_frontend/chakra extern/network_backend/analytical \
       extern/remote_memory_backend/analytical extern/helper/fmt extern/helper/spdlog \
    && git -C /opt/astra-sim rev-parse HEAD > /opt/astra-sim-revision

# Generate BOTH bindings with the container's protoc. No host protobuf ABI leaks.
RUN cd /opt/astra-sim/extern/graph_frontend/chakra/schema/protobuf \
    && protoc --cpp_out=. --python_out=. et_def.proto \
    && cmake -S /opt/astra-sim/build/astra_analytical -B /opt/astra-build \
       -DBUILDTARGET=all -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/astra-build --parallel "${BUILD_JOBS}"

# Only schema, framing utilities and jsonizer are needed for synthetic traces.
# Full Chakra installation pulls in unrelated CUDA trace-link dependencies.
RUN mkdir -p /opt/chakra-python/chakra/schema/protobuf /opt/chakra-python/chakra/src/third_party \
    && cp /opt/astra-sim/extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py /opt/chakra-python/chakra/schema/protobuf/ \
    && cp -r /opt/astra-sim/extern/graph_frontend/chakra/src/third_party/utils /opt/chakra-python/chakra/src/third_party/ \
    && cp -r /opt/astra-sim/extern/graph_frontend/chakra/src/jsonizer /opt/chakra-python/chakra/src/

FROM simulator AS runtime
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/astra-build/bin:$PATH \
    PYTHONPATH=/opt/chakra-python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp
RUN python3 -m venv /opt/venv
COPY scheduler/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY scheduler /workspace/scheduler
WORKDIR /workspace/scheduler
CMD ["python", "-m", "astra.run", "--workload", "prefill-decode"]
