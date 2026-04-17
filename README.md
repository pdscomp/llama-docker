# llama-docker: Router Mode LLM Server with NVIDIA CUDA & AMD Vulkan Support

A Docker-based template for running **llama.cpp llama-server** in router mode with support for NVIDIA CUDA (NVIDIA GPUs) and AMD Vulkan (AMD integrated graphics). Optimized for high-performance inference with 24GB or more of VRAM using **TurboQuant** technology.

> **TurboQuant** is advanced KV cache quantization technology developed primarily by GitHub contributors [TheTom](https://github.com/TheTom) and [SpiritBuun](https://github.com/spiritbuun), enabling efficient long-context inference with minimal quality loss.

## Features

- **GPU-Accelerated Inference**: Full support for NVIDIA CUDA (RTX 3090, RTX 4090, etc.) and AMD Vulkan (Radeon 780M, Strix Halo configurations, and discrete AMD GPUs)
- **Router Mode**: Efficiently manage and load multiple models, serving one at a time with full context window
- **Advanced Quantization**: Built with turboquant support for optimized KV cache handling
- **Long Context**: Up to 262,144 token context window with adaptive KV cache quantization
- **Flexible Configuration**: Simple INI-based model and server configuration
- **Docker Containerization**: Easy deployment and reproducibility

## Tested Hardware

- **NVIDIA**: NVIDIA RTX 3090 (24GB) — primary target platform
- **AMD**: AMD Radeon 780M, Strix Halo configurations, and other AMD integrated/discrete GPUs with sufficient VRAM

> **Note**: AMD configurations have been tested by the author but have not yet been integrated into this repository, plan is to add them once doing further TurboQuant testing on the AMD platform!

## Prerequisites

### 1. Clone and Build llama.cpp

This repository requires a **turboquant-enabled build** of llama.cpp. You have two options:

#### Option A: TheTom's llama-cpp-turboquant fork
```bash
cd llama-build
./rc.fetch-TheTom
./rc.build-TheTom
```

#### Option B: buun's llama-cpp fork (Currently Used by Author)
```bash
cd llama-build
./rc.fetch-buun
./rc.build-buun
```

Both forks support turboquant and work well with the configurations provided. The author is currently using **buun's variant**, which has integrated much of TheTom's work. Both are excellent choices and provide significantly better quality and performance for KV cache quantization.

### 2. Docker & NVIDIA Container Runtime

- **Docker**: [Install Docker](https://docs.docker.com/install/)
- **NVIDIA Container Runtime**: Required for CUDA support
  - [Install NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-container-runtime)
  - Verify with: `docker run --rm --runtime=nvidia nvidia/cuda:13.1.0-base nvidia-smi`

### 3. Models

Place GGUF model files in the `./models` directory. Models are automatically downloaded by the server based on the configuration in `config.ini` using Hugging Face Hub integration.

## Quick Start

### 1. Build the Docker Image

After cloning and building llama.cpp, the Docker image will be available as `llama.cpp:server-cuda-turbo`.

To rebuild or update:
```bash
cd llama-build
./rc.build-TheTom  # or ./rc.build-buun
```

### 2. Configure Models & Server

Edit `config.ini` to specify:
- Models to load (name, Hugging Face repository, quantization)
- Server settings (port, context size, KV cache strategy)
- Performance tuning (GPU layers, parallel slots, etc.)

Example configuration (already provided):
```ini
[*]
host = 0.0.0.0
port = 8080
ctx-size = 262144
cache-type-k = q8_0        # K-cache: full 8-bit quantization
cache-type-v = turbo4      # V-cache: aggressive turbo4 quantization
fit = on
n-gpu-layers = 99
models-max = 1
parallel = 1
```

### 3. Run the Server

```bash
# Start the server in the background
./rc.start

# View logs
docker compose logs -f

# Stop the server
./rc.stop
```

The server will be available at `http://localhost:8080`.

Once the service is running, you can open the built-in llama.cpp web UI at `http://<server-hostname>:8080` to quickly test prompts in a browser (`localhost` if you're running it on the same machine).

### API Access

- **Health Check**: `curl http://localhost:8080/health`
- **Load Model**: `POST /model/load` with model name
- **Inference**: `POST /completion` with prompt and parameters
- **Slots API**: `GET /slots` to inspect active inference slots
- **Full Documentation**: See [llama.cpp API documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)

## Important Changes Since the Original Template

- **`compose.yml`**: No functional changes from the original template. It still runs `llama.cpp:server-cuda-turbo`, exposes `8080:8080`, mounts `./models` and `./config.ini`, and starts with `--models-preset /config.ini`.
- **`config.ini`**:
  - Added/tuned GPU-offload settings: `flash-attn = true`, `fit-target = 256`, and `n-gpu-layers` increased from `99` to `999`.
  - Default startup model moved from `[Qwen3.5-27B]` to `[Qwen3.6-35B-MoE]` (`load-on-startup = true` now set there).
  - MoE preset quantization updated to `Q4_K_M` (`Qwen3.5-35B-MoE` changed from `...Q4_K_XL.gguf` to `...Q4_K_M.gguf`), and a new `[Qwen3.6-35B-MoE]` preset was added.
  - CPU thread tuning guidance changed to prefer leaving `threads`/`threads-batch` commented on asymmetric-core systems.

## Configuration Guide

### Recommended Configurations

Based on testing with a 24GB NVIDIA RTX 3090:

Presets for Qwen3.5-27b (dense), Qwen3.5-35b-MoE, and Qwen3.6-35b-MoE.
Change which one loads on boot by editing bottom of `config.ini` and restarting.

**Best Quality + Performance** (Default):
- **K-cache**: `q8_0` (full 8-bit precision)
- **V-cache**: `turbo4` (4-bit quantization)
- **Quantization**: Q4_K_M (better perplexity than Q4_K_XL which also fits?)
- **Context**: Up to 256K tokens

**Balanced Configurations to Experiment With**:
- `turbo4/turbo4` — More aggressive quantization, increased throughput
- `turbo3/turbo3` — Even more aggressive, lower quality
- `q8_0/turbo3` — Full K-cache precision, aggressive V-cache

Adjust based on your specific use case:
- Prioritizing output quality → Use `q8_0` for K-cache
- Maximizing throughput → Use turbo quantization for both caches
- Limited VRAM → Reduce context size or use more aggressive quantization

### Key Configuration Parameters

| Parameter | Purpose |
|-----------|---------|
| `ctx-size` | Maximum context window size (default: 262144) |
| `cache-type-k` | K-cache quantization: `q8_0`, `q4_0`, `turbo4`, etc. |
| `cache-type-v` | V-cache quantization: `turbo4`, `turbo3`, etc. |
| `kv-unified` | Use shared memory pool for KV cache |
| `flash-attn` | Enable Flash Attention for faster attention kernels on supported builds |
| `fit-target` | VRAM usage target percentage used by `fit = on` |
| `n-gpu-layers` | Number of model layers to offload to GPU (`999` = effectively all layers) |
| `models-max` | Maximum simultaneous models (1 = prevent OOM) |
| `parallel` | Number of concurrent inference slots |
| `mlock` | Lock model in RAM to prevent swapping |
| `mmap` | Memory-map model file for faster loading |
| `sleep-idle-seconds` | Prevent automatic model unloading (-1 = never unload) |

## Project Structure

```
.
├── README.md                 # This file
├── compose.yml             # Docker Compose configuration
├── config.ini              # Server & model configuration
├── rc.start                # Start the server
├── rc.stop                 # Stop the server
├── models/                 # Model storage directory
│   └── .gitkeep
├── llama-build/            # Build scripts and Dockerfile
│   ├── rc.fetch-TheTom     # Clone TheTom's llama.cpp fork
│   ├── rc.build-TheTom     # Build Docker image from TheTom's fork
│   ├── rc.fetch-buun       # Clone buun's llama.cpp fork
│   ├── rc.build-buun       # Build Docker image from buun's fork
│   ├── cuda-turbo.Dockerfile  # Multi-stage Dockerfile for CUDA builds
│   ├── llama-cpp-turboquant/   # (Created by rc.fetch-TheTom)
│   └── buun-llama-cpp/         # (Created by rc.fetch-buun)
└── .gitignore
```

## Usage Examples

### Start the Server and Load a Model

```bash
./rc.start
curl http://localhost:8080/health  # Wait for healthy status
```

The server will load the model specified by `load-on-startup = true` in `config.ini`.

For a quick smoke test from a browser, open `http://<server-hostname>:8080` (or `http://localhost:8080` locally) to use the llama.cpp web portal.

### Generate Completions

```bash
curl http://localhost:8080/completion \
  -d '{
    "prompt": "Once upon a time",
    "n_predict": 100,
    "temperature": 0.7
  }'
```

### Load a Different Model

```bash
curl -X POST http://localhost:8080/model/load \
  -H "Content-Type: application/json" \
  -d '{"name": "Qwen3.5-35B-MoE"}'
```

### Monitor Active Slots

```bash
curl http://localhost:8080/slots
```

## Troubleshooting

### GPU Not Detected

- Verify NVIDIA Container Runtime is installed: `docker run --rm --runtime=nvidia nvidia/cuda:13.1.0-base nvidia-smi`
- Check `CUDA_VISIBLE_DEVICES` in `compose.yml` matches your available GPUs
- Ensure NVIDIA drivers are up-to-date on your host system

### Out of Memory (OOM)

- Reduce `ctx-size` in `config.ini`
- Use more aggressive V-cache quantization (`turbo3` instead of `turbo4`)
- Set `models-max = 1` to prevent multiple models in memory
- Reduce `parallel` to decrease concurrent inference slots

### Model Load Failures

- Verify the Hugging Face repository and file names in `config.ini`
- Ensure sufficient disk space for model downloads
- Check internet connectivity and Hugging Face Hub availability

### Performance Issues

- Reduce `threads` and `threads-batch` if CPU bottlenecked
- Increase `batch-size` and `ubatch-size` for higher throughput
- Disable `mlock` if experiencing swap issues (not recommended)

## Development & Customization

### Building Custom Images

To customize the build for a specific CUDA architecture:

```bash
cd llama-build
docker build \
  -f cuda-turbo.Dockerfile \
  --target server \
  --build-arg CUDA_DOCKER_ARCH=86 \
  -t llama.cpp:server-cuda-turbo-custom \
  ./llama-cpp-turboquant
```

### Adding New Models

Edit `config.ini` and add a new section:

```ini
[MyModel-Name]
hf-repo = username/model-repo-name
hf-file = model-filename.gguf
load-on-startup = false
ctx-size = 131072
```

## Todo

- [ ] Expand AMD Vulkan documentation and configuration examples
- [ ] Test and provide optimized configurations for AMD Radeon 780M and Strix Halo
- [ ] Add Vulkan-specific build scripts and Dockerfile variants
- [ ] Performance benchmarking suite for different quantization modes
- [ ] Extended examples with various inference patterns (streaming, multi-turn chat, etc.)

**Note**: AMD configurations have been tested by the author but will be documented more thoroughly in future versions.

## Contributing

Contributions are welcome! Please submit issues and pull requests for bugs, improvements, or additional configurations.

## License

This repository template is provided as-is. Refer to llama.cpp's original licensing for the core application.

## References

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama.cpp Server Documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
- [TheTom's llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
- [buun's llama-cpp](https://github.com/spiritbuun/buun-llama-cpp)
- [NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-container-runtime)

---

**Author**: pdscomp

**Last Updated**: April 2026
