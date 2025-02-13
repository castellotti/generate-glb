import os
import sys
import argparse
from threading import Thread
import tempfile
import numpy as np
import trimesh
from queue import Empty
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from trimesh.exchange.gltf import export_glb

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
    return parser.parse_args()

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

    # Load the mesh
    try:
        mesh = trimesh.load_mesh(temp_file, file_type='obj')
    except Exception as e:
        print(f"Error loading mesh: {e}")
        if verbose:
            print("Full mesh text:")
            print(mesh_text)
        os.unlink(temp_file)
        return None

    # Get vertex coordinates
    vertices = mesh.vertices
    if len(vertices) == 0:
        print("Error: No vertices found in the mesh")
        os.unlink(temp_file)
        return None

    y_values = vertices[:, 1]  # Y-axis values

    if verbose:
        print(f"\nNumber of vertices: {len(vertices)}")
        print(f"Number of faces: {len(mesh.faces)}")

    # Normalize Y values to range [0, 1] for color mapping
    y_min, y_max = y_values.min(), y_values.max()
    if y_min == y_max:
        y_normalized = np.zeros_like(y_values)
    else:
        y_normalized = (y_values - y_min) / (y_max - y_min)

    # Generate colors: Map normalized Y values to RGB gradient
    colors = np.zeros((len(vertices), 4))  # RGBA
    colors[:, 0] = y_normalized  # Red channel
    colors[:, 2] = 1 - y_normalized  # Blue channel
    colors[:, 3] = 1.0  # Alpha channel (fully opaque)

    # Attach colors to mesh vertices
    mesh.visual.vertex_colors = colors

    # Export to GLB format
    try:
        with open(output_path, "wb") as f:
            f.write(export_glb(mesh))
    except Exception as e:
        print(f"Error saving GLB file: {e}")
        os.unlink(temp_file)
        return None

    # Clean up temporary file
    os.unlink(temp_file)
    return output_path

def generate_mesh(prompt, temperature, max_new_tokens, timeout=300.0, verbose=False):
    """
    Generate a 3D mesh using the LLaMA-Mesh model.
    """
    try:
        # Set up model and tokenizer
        if verbose:
            print("Loading model and tokenizer...")

        model_path = "Zhengyi/LLaMA-Mesh"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto"
        )

        # Prepare the attention mask
        conversation = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt")
        attention_mask = inputs.ne(tokenizer.pad_token_id).to(model.device)
        inputs = inputs.to(model.device)

        # Set up streamer with longer timeout
        streamer = TextIteratorStreamer(
            tokenizer,
            timeout=timeout,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Set up generation parameters
        generate_kwargs = {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        }

        if verbose:
            print("Starting generation...")

        # Generate response
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        # Collect response
        response = ""
        buffer = []
        number_buffer = ""

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

        def print_buffer():
            """Print buffer contents if it forms a complete vertex or face"""
            if not buffer:
                return

            # Check if we have exactly 3 coordinates for a vertex
            if len(buffer) == 3 and all(isinstance(x, float) for x in buffer):
                print(f"v {' '.join(str(x) for x in buffer)}")
                buffer.clear()
            # Check if we have at least 3 indices for a face
            elif len(buffer) >= 3 and all(x.is_integer() for x in buffer):
                print(f"f {' '.join(str(int(x)) for x in buffer)}")
                buffer.clear()

        try:
            for text in streamer:
                response += text
                if verbose:
                    # Process each character
                    for char in text:
                        if char.isspace():
                            if number_buffer:
                                num = try_parse_number(number_buffer)
                                if num is not None:
                                    buffer.append(num)
                                    print_buffer()
                                number_buffer = ""
                        elif char.isdigit() or char == '.' or char == '-':
                            number_buffer += char
                        elif char == 'v':
                            if buffer:
                                print_buffer()
                            buffer = []
                            number_buffer = ""
                        elif char == 'f':
                            if buffer:
                                print_buffer()
                            buffer = []
                            number_buffer = ""

                    # Handle any remaining complete number in the buffer
                    if number_buffer:
                        num = try_parse_number(number_buffer)
                        if num is not None:
                            buffer.append(num)
                            print_buffer()
        except Empty:
            print(f"Warning: Generation timed out after {timeout} seconds")
            if response:
                print("Using partial response...")
            else:
                raise Exception("No response generated before timeout")

        return response

    except Exception as e:
        print(f"Error during mesh generation: {e}")
        raise

def main():
    # Parse command line arguments
    args = parse_arguments()

    print(f"Generating 3D mesh for prompt: {args.prompt}")
    print(f"Using temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")

    try:
        # Generate the mesh
        mesh_text = generate_mesh(
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

        # Extract the OBJ content
        mesh_content = mesh_text[obj_start:]

        # Apply gradient color and save as GLB
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
