ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION} AS build

RUN apt-get update && \
    apt-get install -y build-essential cmake git libvulkan-dev vulkan-tools spirv-tools python3 python3-pip

WORKDIR /app
COPY . .

RUN cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_VULKAN=ON \
    -DGGML_VULKAN_SHADERGEN=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    . && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \; && \
    mkdir -p /app/full && \
    cp build/bin/* /app/full && \
    cp *.py /app/full && \
    cp -r gguf-py /app/full && \
    cp -r requirements /app/full && \
    cp requirements.txt /app/full && \
    cp .devops/tools.sh /app/full/tools.sh

FROM ubuntu:${UBUNTU_VERSION} AS base

RUN apt-get update \
    && apt-get install -y libgomp1 curl vulkan-tools \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/*

COPY --from=build /app/lib/ /app

FROM base AS full
COPY --from=build /app/full /app
WORKDIR /app
RUN apt-get update && apt-get install -y git python3 python3-pip \
    && pip install --break-system-packages -r requirements.txt \
    && apt clean -y && rm -rf /tmp/*
ENTRYPOINT ["/app/tools.sh"]

FROM base AS light
COPY --from=build /app/full/llama-cli /app/full/llama-completion /app
WORKDIR /app
ENTRYPOINT ["/app/llama-cli"]

FROM base AS server
ENV LLAMA_ARG_HOST=0.0.0.0
COPY --from=build /app/full/llama-server /app
WORKDIR /app
HEALTHCHECK CMD ["curl", "-f", "http://localhost:8080/health"]
ENTRYPOINT ["/app/llama-server"]
