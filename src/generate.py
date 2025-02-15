#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py

Copyright (c) 2025 Steven Castellotti
Licensed under the MIT License. See LICENSE file for details.

Description:
    This script generates 3D meshes using the LLaMA-Mesh model and exports them in GLB format.
    It utilizes `trimesh` for mesh processing and `psutil` for system resource monitoring.
    Supported backends include PyTorch, llama.cpp, and Ollama.
    Hardware acceleration is supported via NVIDIA CUDA or Apple Metal.

Author: Steven Castellotti
Version: 1.0.0
Date: 2025-02-14
"""

import argparse
import io
import numpy as np
import os
import psutil
import sys
import tempfile
import time
import trimesh
from contextlib import contextmanager
from queue import Empty
from threading import Thread
from trimesh.exchange.gltf import export_glb

# Base configuration
BASE_NAME = "LLaMA-Mesh"
TRANSFORMERS_AUTHOR = "Zhengyi"
HF_DOMAIN = "hf.co"
GGUF_AUTHOR = "bartowski"

# Model configuration
TRANSFORMERS_MODEL_REPO = f"{TRANSFORMERS_AUTHOR}/{BASE_NAME}"
DEFAULT_MODEL_REPO = f"{GGUF_AUTHOR}/{BASE_NAME}-GGUF"
MODEL_FILE_BASE = f"{BASE_NAME}-"
HF_BASE_PATH = f"{HF_DOMAIN}/{DEFAULT_MODEL_REPO}"

MODEL_VARIANTS = {
    'f16': 'Full F16 weights',
    'q8_0': 'Extremely high quality',
    'q6_k_l': 'Very high quality with Q8_0 embed/output weights',
    'q6_k': 'Very high quality',
    'q5_k_l': 'High quality with Q8_0 embed/output weights',
    'q5_k_m': 'High quality',
    'q5_k_s': 'High quality, smaller',
    'q4_k_l': 'Good quality with Q8_0 embed/output weights',
    'q4_k_m': 'Good quality, default recommendation',
    'q4_k_s': 'Good quality, space optimized',
    'q3_k_xl': 'Lower quality with Q8_0 embed/output weights',
    'q3_k_l': 'Lower quality',
    'q3_k_m': 'Low quality',
    'q3_k_s': 'Low quality, not recommended',
    'q2_k_l': 'Very low quality with Q8_0 embed/output weights',
    'q2_k': 'Very low quality',
    'iq4_xs': 'Decent quality, very space efficient',
    'iq3_m': 'Medium-low quality',
    'iq3_xs': 'Lower quality',
    'iq2_m': 'Relatively low quality, SOTA techniques'
}

# Create case-insensitive variant lookup
VARIANT_LOOKUP = {k.lower(): k for k in MODEL_VARIANTS.keys()}

DEFAULT_VARIANT = 'q4_k_m'

def get_variant(variant_input):
    """Get the canonical variant name, handling case-insensitivity."""
    lookup_key = variant_input.lower()
    return VARIANT_LOOKUP.get(lookup_key)

def list_model_variants():
    print("\nAvailable model variants:")
    print(f"{'Variant':<10} Description")
    print("-" * 60)
    for variant, description in MODEL_VARIANTS.items():
        print(f"{variant:<10} {description}")

def get_model_filename(variant):
    """Generate the GGUF filename for a given variant."""
    return f"{MODEL_FILE_BASE}{variant.upper()}.gguf"

def get_ollama_model_path(variant):
    """Convert variant to Ollama model path."""
    return f"{HF_BASE_PATH}:{variant.upper()}"

def ensure_model_downloaded(repo_id, variant):
    """Download model from HuggingFace if not already present."""
    from huggingface_hub import hf_hub_download

    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")

    filename = get_model_filename(variant)
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Using model at: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to suppress stdout and stderr output.
    """
    # Save the original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Create null devices
    null_stdout = io.StringIO()
    null_stderr = io.StringIO()

    try:
        # Redirect stdout/stderr to null devices
        sys.stdout = null_stdout
        sys.stderr = null_stderr
        yield
    finally:
        # Restore original stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate 3D meshes using LLaMA-Mesh from command line')
    parser.add_argument('prompt', type=str, help='Prompt for generating the 3D mesh')
    parser.add_argument('--temperature', type=float, default=0.95,
                        help='Temperature for generation (default: 0.95)')
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate (default: 4096)')
    parser.add_argument('--output', type=str, default='output.glb',
                        help='Output filename for the GLB file (default: output.glb)')
    parser.add_argument('--verbose', action='store_true',
                        help='Display vertices and faces as they are generated')
    parser.add_argument('--timeout', type=float, default=900.0,
                        help='Timeout in seconds for generation (default: 900.0)')
    parser.add_argument('--backend', type=str, choices=['transformers', 'llama_cpp', 'ollama'],
                        default='transformers', help='Backend to use for generation')
    parser.add_argument('--model-path', type=str,
                        help='Path to model file (if not specified, will download from HuggingFace)')
    parser.add_argument('--ollama-host', type=str, default='http://localhost:11434',
                        help='Host address for Ollama backend')
    parser.add_argument('--variant', type=str, default=DEFAULT_VARIANT,
                        help=f'Model variant to use (default: {DEFAULT_VARIANT})')
    parser.add_argument('--list-variants', action='store_true',
                        help='List available model variants and exit')

    args = parser.parse_args()

    if args.list_variants:
        list_model_variants()
        sys.exit(0)

    # Handle case-insensitive variant lookup
    canonical_variant = get_variant(args.variant)
    if canonical_variant is None:
        print(f"Error: Unknown model variant '{args.variant}'")
        list_model_variants()
        sys.exit(1)
    args.variant = canonical_variant

    return args

class TransformersBackend:
    def __init__(self, stats=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL_REPO)
        self.model = AutoModelForCausalLM.from_pretrained(
            TRANSFORMERS_MODEL_REPO,
            device_map="auto",
            torch_dtype="auto"
        )
        if stats:
            stats.model_load_time = time.time() - start_time

    def generate(self, prompt, temperature, max_new_tokens, timeout):
        from transformers import TextIteratorStreamer
        conversation = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(conversation, return_tensors="pt")
        attention_mask = inputs.ne(self.tokenizer.pad_token_id).to(self.model.device)
        inputs = inputs.to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=timeout,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generate_kwargs = {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        }

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        return streamer

class LlamaCppBackend:
    def __init__(self, model_path=None, variant=DEFAULT_VARIANT, stats=None):
        import llama_cpp

        start_time = time.time()
        if model_path is None:
            model_path = ensure_model_downloaded(DEFAULT_MODEL_REPO, variant)

        # Use context manager to suppress output during model loading
        with suppress_stdout_stderr():
            self.llm = llama_cpp.Llama(
                model_path=str(model_path),
                n_gpu_layers=-1,
                seed=1337,
                n_ctx=4096,
            )
        if stats:
            stats.model_load_time = time.time() - start_time

    def generate(self, prompt, temperature, max_new_tokens, timeout, verbose=False):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can generate 3D obj files."},
            {"role": "user", "content": prompt}
        ]

        # Only suppress output if verbose mode is disabled
        if not verbose:
            with suppress_stdout_stderr():
                return self.llm.create_chat_completion(
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_new_tokens
                )
        else:
            return self.llm.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_new_tokens
            )

class OllamaBackend:
    def __init__(self, host, variant=DEFAULT_VARIANT, stats=None):
        from ollama import Client
        self.client = Client(host=host)

        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

        self.model_name = get_ollama_model_path(variant)
        print(f"Using Ollama model: {self.model_name}")

        # Pull the model into Ollama
        print(f"Pulling model into Ollama: {self.model_name}")
        try:
            start_time = time.time()
            self.client.pull(self.model_name)
            if stats:
                stats.model_load_time = time.time() - start_time
        except Exception as e:
            print(f"Error pulling model: {e}")
            raise

    def generate(self, prompt, temperature, max_new_tokens, timeout):
        # Enhanced template with explicit mesh generation instructions
        template = """<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that can generate 3D obj files. Generate a complete .obj format 3D mesh in response to the user's request. Start the response with vertex (v) definitions followed by face (f) definitions.<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Here is the 3D mesh in .obj format:"""

        return self.client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=True,
            template=template,
            options={
                "temperature": temperature,
                "num_predict": max_new_tokens,
                "stop": ["<|eot_id|>"]  # Add explicit stop token
            }
        )

def try_parse_number(num_str):
    """Attempt to parse a number string, return None if incomplete"""
    try:
        # Check if the string ends with a decimal point
        if num_str.endswith('.'):
            return None
        # Check if the string is just a minus sign
        if num_str == '-':
            return None
        # Try to convert to float
        return float(num_str)
    except ValueError:
        return None

def process_buffer(buffer, verbose=False):
    """Process and print buffer contents if it forms a complete vertex or face"""
    if not buffer:
        return

    # Check if we have exactly 3 coordinates for a vertex
    if len(buffer) == 3 and all(isinstance(x, float) for x in buffer):
        if verbose:
            print(f"v {' '.join(str(x) for x in buffer)}")
        buffer.clear()
    # Check if we have at least 3 indices for a face
    elif len(buffer) >= 3 and all(x.is_integer() for x in buffer):
        if verbose:
            print(f"f {' '.join(str(int(x)) for x in buffer)}")
        buffer.clear()

def process_stream(stream, is_ollama=False, verbose=False):
    """Process the stream with detailed vertex/face parsing for verbose output"""
    response = ""
    buffer = []
    number_buffer = ""

    try:
        for chunk in stream:
            # Extract content based on backend type
            if is_ollama:
                content = chunk["response"]
            else:
                if isinstance(chunk, dict):
                    delta = chunk["choices"][0]["delta"]
                    if "content" not in delta:
                        continue
                    content = delta["content"]
                else:
                    content = chunk

            response += content

            if verbose:
                # Process each character for detailed vertex/face parsing
                for char in content:
                    if char.isspace():
                        if number_buffer:
                            num = try_parse_number(number_buffer)
                            if num is not None:
                                buffer.append(num)
                                process_buffer(buffer, verbose)
                            number_buffer = ""
                    elif char.isdigit() or char == '.' or char == '-':
                        number_buffer += char
                    elif char == 'v':
                        if buffer:
                            process_buffer(buffer, verbose)
                        buffer = []
                        number_buffer = ""
                    elif char == 'f':
                        if buffer:
                            process_buffer(buffer, verbose)
                        buffer = []
                        number_buffer = ""

                # Handle any remaining complete number in the buffer
                if number_buffer:
                    num = try_parse_number(number_buffer)
                    if num is not None:
                        buffer.append(num)
                        process_buffer(buffer, verbose)

            yield content

    except Empty:
        print("Warning: Generation timed out")

    if buffer and verbose:
        process_buffer(buffer, verbose)

    return response

class PerformanceStats:
    def __init__(self):
        self.start_time = time.time()
        self.generation_time = 0
        self.export_time = 0
        self.total_time = 0
        self.peak_memory = 0
        self.initial_memory = self._get_memory_usage()
        self.model_load_time = 0  # Combined time for downloading and loading model

    @staticmethod
    def _get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def update_memory(self):
        """Update peak memory usage"""
        current = self._get_memory_usage()
        self.peak_memory = max(self.peak_memory, current)

    @staticmethod
    def get_gpu_stats():
        """Try to get GPU statistics if available"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024 * 1024),
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024 * 1024),
                    'gpu_device': torch.cuda.get_device_name()
                }
        except ImportError:
            pass
        return None

    def print_report(self):
        """Print performance statistics"""
        print("\nPerformance Statistics:")
        print("-" * 50)
        print(f"Total Processing Time: {self.total_time:.2f} seconds")
        print(f"├─ Model Load Time: {self.model_load_time:.2f} seconds")
        print(f"├─ Generation Time: {self.generation_time:.2f} seconds")
        print(f"└─ Export Time: {self.export_time:.2f} seconds")
        print(f"\nMemory Usage:")
        print(f"├─ Initial: {self.initial_memory:.1f} MB")
        print(f"├─ Peak: {self.peak_memory:.1f} MB")
        print(f"└─ Delta: {(self.peak_memory - self.initial_memory):.1f} MB")

        cpu_percent = psutil.cpu_percent(interval=0.1)
        print(f"\nCPU Usage: {cpu_percent}%")

        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            print(f"\nGPU Statistics:")
            print(f"├─ Device: {gpu_stats['gpu_device']}")
            print(f"├─ Allocated Memory: {gpu_stats['gpu_memory_allocated']:.1f} MB")
            print(f"└─ Reserved Memory: {gpu_stats['gpu_memory_reserved']:.1f} MB")

@contextmanager
def timer(stats_obj, timer_name):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if timer_name == 'generation':
            stats_obj.generation_time = elapsed
        elif timer_name == 'export':
            stats_obj.export_time = elapsed
        stats_obj.update_memory()

def generate_mesh(backend, prompt, temperature, max_new_tokens, timeout=900.0, verbose=False, stats=None):
    """Generate a 3D mesh using the selected backend."""
    try:
        if verbose:
            print(f"Generating mesh using {backend.__class__.__name__}")

        with timer(stats, 'generation'):
            stream = backend.generate(prompt, temperature, max_new_tokens, timeout)
            is_ollama = isinstance(backend, OllamaBackend)

            response = ""
            for content in process_stream(stream, is_ollama, verbose):
                response += content
                if stats:
                    stats.update_memory()

        return response

    except Exception as e:
        print(f"Error during mesh generation: {e}")
        raise

def apply_gradient_color(mesh_text, output_path, verbose=False, stats=None):
    """Apply a gradient color to the mesh vertices based on the Y-axis and save as GLB."""
    with timer(stats, 'export'):
        # Create temporary OBJ file
        temp_file = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
        with open(temp_file, "w") as f:
            f.write(mesh_text)

        if verbose:
            print("\nMesh content:")
            for line in mesh_text.split('\n'):
                if line.startswith('v ') or line.startswith('f '):
                    print(line)

        try:
            mesh = trimesh.load_mesh(temp_file, file_type='obj')
        except Exception as e:
            print(f"Error loading mesh: {e}")
            if verbose:
                print("Full mesh text:")
                print(mesh_text)
            os.unlink(temp_file)
            return None

        vertices = mesh.vertices
        if len(vertices) == 0:
            print("Error: No vertices found in the mesh")
            os.unlink(temp_file)
            return None

        if verbose:
            print(f"\nNumber of vertices: {len(vertices)}")
            print(f"Number of faces: {len(mesh.faces)}")

        # Apply gradient coloring
        y_values = vertices[:, 1]
        y_min, y_max = y_values.min(), y_values.max()
        y_normalized = np.zeros_like(y_values) if y_min == y_max else (y_values - y_min) / (y_max - y_min)

        colors = np.zeros((len(vertices), 4))
        colors[:, 0] = y_normalized  # Red channel
        colors[:, 2] = 1 - y_normalized  # Blue channel
        colors[:, 3] = 1.0  # Alpha channel

        mesh.visual.vertex_colors = colors

        try:
            with open(output_path, "wb") as f:
                f.write(export_glb(mesh))
        except Exception as e:
            print(f"Error saving GLB file: {e}")
            os.unlink(temp_file)
            return None

        os.unlink(temp_file)
        return output_path

def main():
    args = parse_arguments()
    print(f"Generating 3D mesh for prompt: {args.prompt}")
    print(f"Using temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Using backend: {args.backend}")
    print(f"Using model variant: {args.variant}")

    # Initialize performance tracking
    stats = PerformanceStats()

    try:
        # Initialize the selected backend
        if args.backend == 'llama_cpp':
            backend = LlamaCppBackend(args.model_path, args.variant, stats)
        elif args.backend == 'ollama':
            backend = OllamaBackend(args.ollama_host, args.variant, stats)
        else:
            # default: 'transformers'
            backend = TransformersBackend(stats)

        # Generate the mesh
        mesh_text = generate_mesh(
            backend,
            args.prompt,
            args.temperature,
            args.max_tokens,
            timeout=args.timeout,
            verbose=args.verbose,
            stats=stats
        )

        # Look for OBJ content in the response
        obj_start = mesh_text.find("v ")
        if obj_start == -1:
            print("No valid mesh found in the response")
            print("Response:", mesh_text)
            return 1

        # Extract the OBJ content and apply gradient color
        mesh_content = mesh_text[obj_start:]
        output_path = apply_gradient_color(mesh_content, args.output, args.verbose, stats)

        # Calculate total time and print statistics
        stats.total_time = time.time() - stats.start_time
        stats.print_report()

        if output_path:
            print(f"\nMesh saved to: {output_path}")
            return 0
        return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.exit(main())
