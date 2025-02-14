# generate-glb

<div style="text-align:center;">
  <img src="docs/images/wooden_hammer.png" alt="Wooden Hammer" width="360"/>
</div>

## Features
- Command line tool to generate 3D models in GLB format
- Supports hardware acceleration:
  - NVIDIA CUDA
  - Apple Metal
  - Ollama (local and remote servers)

### Tips
- For Apple Silicon system, running locally with Metal support is available via the `llama_cpp` or `ollama` backends (with Ollama running on the local system or a remote server)
- For NVIDIA systems, running locally with acceleration is available via the `llama_cpp` or `ollama` backends, or through Docker (see [Containers for Deep Learning Frameworks](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html)) 
- For Docker containers, if NVIDIA CUDA is not available, acceleration is still possible through Ollama (running on the local system or a remote server)

## Instructions

### Setup
```shell
git clone https://github.com/yourusername/generate-glb.git
cd generate-glb
```

#### Local

Instantiate a Python virtual environment

```pyenv virtualenv 3.11.6 generate-glb-env```

Activate virtual environment

```pyenv activate generate-glb-env```

Install requirements

```pip install -r docker/requirements/base.txt```

**Optional**: Install hardware acceleration

##### NVIDIA CUDA

```pip install -r docker/requirements/cuda.txt```

##### Apple Metal

```pip install -r docker/requirements/metal.txt```

##### Llama.cpp Backend (CPU only)

```pip install -r docker/requirements/cpu.txt```

#### Docker

##### Local (CPU only)
```docker compose -f docker/docker-compose.yml run generate-glb "Create a 3D model of a sword"```

##### Local (NVIDIA CUDA)
```shell
ACCELERATION=cuda docker compose \
  -f docker/docker-compose.yml \
  run --gpus all generate-glb \
  "Create a 3D model of a sword"
```

##### Network (local Ollama)
```shell
docker compose \
  -f docker/docker-compose.yml \
  run generate-glb \
  --backend ollama \
  --ollama-host http://host.docker.internal:11434 \
  "Create a 3D model of a sword"
```

##### Network (remote Ollama server)
```shell
docker compose \
  -f docker/docker-compose.yml \
  run generate-glb \
  --backend ollama \
  --ollama-host http://192.168.1.100:11434 \
  "Create a 3D model of a sword"
```

### Usage
Command: `python src/generate.py --help`
```
usage: generate.py [-h] [--temperature TEMPERATURE] [--max-tokens MAX_TOKENS] [--output OUTPUT] [--verbose]
                   [--timeout TIMEOUT] [--backend {transformers,llama_cpp,ollama}] [--model-path MODEL_PATH]
                   [--ollama-host OLLAMA_HOST] [--variant VARIANT] [--list-variants]
                   prompt

Generate 3D meshes using LLaMA-Mesh from command line

positional arguments:
  prompt                Prompt for generating the 3D mesh

options:
  -h, --help            show this help message and exit
  --temperature TEMPERATURE
                        Temperature for generation (default: 0.95)
  --max-tokens MAX_TOKENS
                        Maximum number of tokens to generate (default: 4096)
  --output OUTPUT       Output filename for the GLB file (default: output.glb)
  --verbose             Display vertices and faces as they are generated
  --timeout TIMEOUT     Timeout in seconds for generation (default: 900.0)
  --backend {transformers,llama_cpp,ollama}
                        Backend to use for generation
  --model-path MODEL_PATH
                        Path to model file (if not specified, will download from HuggingFace)
  --ollama-host OLLAMA_HOST
                        Host address for Ollama backend
  --variant VARIANT     Model variant to use (default: q4_k_m)
  --list-variants       List available model variants and exit
```

### Models
Currently supported model variants ([source](https://huggingface.co/bartowski/LLaMA-Mesh-GGUF))
```
Available model variants:
Variant    Description
------------------------------------------------------------
f16        Full F16 weights
q8_0       Extremely high quality
q6_k_l     Very high quality with Q8_0 embed/output weights
q6_k       Very high quality
q5_k_l     High quality with Q8_0 embed/output weights
q5_k_m     High quality
q5_k_s     High quality, smaller
q4_k_l     Good quality with Q8_0 embed/output weights
q4_k_m     Good quality, default recommendation
q4_k_s     Good quality, space optimized
q3_k_xl    Lower quality with Q8_0 embed/output weights
q3_k_l     Lower quality
q3_k_m     Low quality
q3_k_s     Low quality, not recommended
q2_k_l     Very low quality with Q8_0 embed/output weights
q2_k       Very low quality
iq4_xs     Decent quality, very space efficient
iq3_m      Medium-low quality
iq3_xs     Lower quality
iq2_m      Relatively low quality, SOTA techniques
```

## Example

### Command
```shell
python src/generate.py \
  --temperature 0.95 \
  --max-tokens 4096 \
  --output sword.glb \
  "Create a 3D model of a sword"
```

### Model

<div style="text-align:center;">
  <img src="docs/images/sword.png" alt="Sword" width="360"/>
</div>

### Output
```
Generating 3D mesh for prompt: Create a 3D model of a sword
Using temperature: 0.95
Max tokens: 4096
Using backend: transformers
Using model variant: q4_k_m
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.50s/it]

Performance Statistics:
--------------------------------------------------
Total Processing Time: 136.96 seconds
├─ Model Load Time: 6.85 seconds
├─ Generation Time: 128.37 seconds
└─ Export Time: 0.00 seconds

Memory Usage:
├─ Initial: 51.9 MB
├─ Peak: 6140.8 MB
└─ Delta: 6088.9 MB

CPU Usage: 0.0%

Mesh saved to: sword.glb
```

## Credits & Acknowledgments
- This project is based on [LLaMA-Mesh](https://github.com/nv-tlabs/LLaMA-Mesh) and incorporates code licensed under the NVIDIA License.
- Original work by **NVIDIA Toronto AI Lab** is licensed under the **NVIDIA License** (see `LICENSE_NVIDIA`).
- This project also derives portions from [meshgen](https://github.com/huggingface/meshgen) (Copyright (c) 2024 Hugging Face) and licensed under the MIT License.
- This software also uses the Llama model, which is governed by the **Llama Community License** (see `LICENSE_LLAMA`).
- All modifications and additional code contributions in this repository are licensed under the **MIT License** (Copyright (c) 2025 Steven Castellotti).
