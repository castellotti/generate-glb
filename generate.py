import os
import sys
import argparse
import tempfile
import numpy as np
import trimesh
from threading import Thread
from queue import Empty
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

# Simplified model variants with just quant type and description
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

def apply_gradient_color(mesh_text, output_path, verbose=False):
    """
    Apply a gradient color to the mesh vertices based on the Y-axis and save as GLB.
    """
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

class TransformersBackend:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL_REPO)
        self.model = AutoModelForCausalLM.from_pretrained(
            TRANSFORMERS_MODEL_REPO,
            device_map="auto",
            torch_dtype="auto"
        )

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
    def __init__(self, model_path=None, variant=DEFAULT_VARIANT):
        import llama_cpp

        if model_path is None:
            model_path = ensure_model_downloaded(DEFAULT_MODEL_REPO, variant)

        self.llm = llama_cpp.Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,
            seed=1337,
            n_ctx=4096,
        )

    def generate(self, prompt, temperature, max_new_tokens, timeout):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can generate 3D obj files."},
            {"role": "user", "content": prompt}
        ]
        return self.llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_new_tokens
        )

class OllamaBackend:
    def __init__(self, host, variant=DEFAULT_VARIANT):
        from ollama import Client
        self.client = Client(host=host)

        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

        self.model_name = get_ollama_model_path(variant)
        print(f"Using Ollama model: {self.model_name}")

        # Pull the model into Ollama
        print(f"Pulling model into Ollama: {self.model_name}")
        try:
            self.client.pull(self.model_name)
        except Exception as e:
            print(f"Error pulling model: {e}")
            raise

    def generate(self, prompt, temperature, max_new_tokens, timeout):
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant that can generate 3D obj files.<|eot_id|><|start_header_id|>user<|end_header_id|>
        {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        return self.client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=True,
            template=template,
            options={"temperature": temperature}
        )

def process_stream(stream, is_ollama=False):
    response = ""
    try:
        for chunk in stream:
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
            yield content
    except Empty:
        print("Warning: Generation timed out")
    return response

def generate_mesh(backend, prompt, temperature, max_new_tokens, timeout=300.0, verbose=False):
    """Generate a 3D mesh using the selected backend."""
    try:
        if verbose:
            print(f"Generating mesh using {backend.__class__.__name__}")

        stream = backend.generate(prompt, temperature, max_new_tokens, timeout)
        is_ollama = isinstance(backend, OllamaBackend)

        response = ""
        for content in process_stream(stream, is_ollama):
            response += content
            if verbose:
                if content.strip():
                    print(content.strip())

        return response

    except Exception as e:
        print(f"Error during mesh generation: {e}")
        raise

def main():
    args = parse_arguments()
    print(f"Generating 3D mesh for prompt: {args.prompt}")
    print(f"Using temperature: {args.temperature}")
    print(f"Using backend: {args.backend}")
    print(f"Using model variant: {args.variant}")

    try:
        # Initialize the selected backend
        if args.backend == 'transformers':
            backend = TransformersBackend()
        elif args.backend == 'llama_cpp':
            backend = LlamaCppBackend(args.model_path, args.variant)
        elif args.backend == 'ollama':
            backend = OllamaBackend(args.ollama_host, args.variant)

        # Generate the mesh
        mesh_text = generate_mesh(
            backend,
            args.prompt,
            args.temperature,
            args.max_tokens,
            timeout=args.timeout,
            verbose=args.verbose
        )

        # Look for OBJ content in the response
        obj_start = mesh_text.find("v ")
        if obj_start == -1:
            print("No valid mesh found in the response")
            print("Response:", mesh_text)
            return 1

        # Extract the OBJ content and apply gradient color
        mesh_content = mesh_text[obj_start:]
        output_path = apply_gradient_color(mesh_content, args.output, args.verbose)

        if output_path:
            print(f"Mesh saved to: {output_path}")
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
