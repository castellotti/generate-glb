# styx-enchant

<div style="text-align:center;">
  <img src="docs/images/wooden_hammer.png" alt="Wooden Hammer" width="360"/>
</div>

## Instructions

### Setup

Instantiate a Python virtual environment

```pyenv virtualenv 3.11.6 styx-enchant-env```

Activate virtual environment

```pyenv activate styx-enchant-env```

Install requirements

```pip install -r requirements.txt```

### Usage
```shell
usage: generate.py [-h] [--temperature TEMPERATURE] [--max-tokens MAX_TOKENS] [--output OUTPUT] [--verbose]
                   [--timeout TIMEOUT]
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
```

## Example

### Command
```shell
python generate.py "Create a 3D model of a sword" \
  --temperature 0.95 \
  --max-tokens 4096 \
  --output sword.glb \
  --verbose \
  --timeout 900
```

### Model

<div style="text-align:center;">
  <img src="docs/images/sword.png" alt="Sword" width="360"/>
</div>

### Output
```shell
Generating 3D mesh for prompt: Create a 3D model of a sword
Using temperature: 0.95
Max tokens: 4096
Loading model and tokenizer...
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.20it/s]
Starting generation...
v 23.0 16.0 29.0
v 23.0 16.0 33.0
v 23.0 19.0 29.0
v 23.0 19.0 33.0
v 24.0 19.0 30.0
v 24.0 19.0 32.0
v 24.0 21.0 30.0
v 24.0 21.0 32.0
v 26.0 21.0 31.0
v 26.0 58.0 31.0
v 28.0 16.0 30.0
v 28.0 16.0 32.0
v 29.0 1.0 30.0
v 29.0 1.0 31.0
v 29.0 1.0 32.0
v 29.0 3.0 30.0
v 29.0 3.0 32.0
v 29.0 4.0 30.0
v 29.0 4.0 32.0
v 30.0 0.0 31.0
v 31.0 0.0 30.0
v 31.0 0.0 31.0
v 31.0 0.0 32.0
v 31.0 1.0 29.0
v 31.0 1.0 30.0
v 31.0 1.0 32.0
v 31.0 1.0 33.0
v 31.0 3.0 29.0
v 31.0 3.0 33.0
v 31.0 4.0 29.0
v 31.0 4.0 33.0
v 31.0 16.0 27.0
v 31.0 16.0 28.0
v 31.0 16.0 34.0
v 31.0 16.0 35.0
v 31.0 19.0 27.0
v 31.0 19.0 28.0
v 31.0 19.0 34.0
v 31.0 19.0 35.0
v 31.0 21.0 28.0
v 31.0 21.0 29.0
v 31.0 21.0 33.0
v 31.0 21.0 34.0
v 31.0 58.0 29.0
v 31.0 58.0 33.0
v 31.0 63.0 31.0
v 32.0 0.0 31.0
v 33.0 1.0 30.0
v 33.0 1.0 31.0
v 33.0 1.0 32.0
v 33.0 3.0 30.0
v 33.0 3.0 32.0
v 33.0 4.0 30.0
v 33.0 4.0 32.0
v 34.0 16.0 30.0
v 34.0 16.0 32.0
v 36.0 21.0 31.0
v 36.0 58.0 31.0
v 38.0 19.0 30.0
v 38.0 19.0 32.0
v 38.0 21.0 30.0
v 38.0 21.0 32.0
v 39.0 16.0 29.0
v 39.0 16.0 33.0
v 39.0 19.0 29.0
v 39.0 19.0 33.0
v 1.0 2.0 3.0
v 1.0 11.0 2.0
v 1.0 3.0 36.0
v 1.0 33.0 11.0
v 1.0 32.0 33.0
v 1.0 36.0 32.0
v 2.0 4.0 3.0
v 2.0 35.0 4.0
v 2.0 11.0 12.0
v 2.0 12.0 35.0
v 3.0 4.0 6.0
v 3.0 6.0 5.0
v 3.0 5.0 36.0
v 4.0 38.0 6.0
v 4.0 35.0 39.0
v 4.0 39.0 38.0
v 5.0 6.0 8.0
v 5.0 8.0 7.0
v 5.0 7.0 37.0
v 5.0 37.0 36.0
v 6.0 43.0 8.0
v 6.0 38.0 43.0
v 7.0 8.0 9.0
v 7.0 9.0 40.0
v 7.0 40.0 37.0
v 8.0 42.0 9.0
v 8.0 43.0 42.0
v 9.0 10.0 41.0
v 9.0 45.0 10.0
v 9.0 41.0 40.0
v 9.0 42.0 45.0
v 10.0 44.0 41.0
v 10.0 46.0 44.0
v 10.0 45.0 46.0
v 11.0 18.0 12.0
v 11.0 30.0 18.0
v 11.0 33.0 30.0
v 12.0 18.0 19.0
v 12.0 19.0 34.0
v 12.0 34.0 35.0
v 13.0 14.0 15.0
v 13.0 25.0 14.0
v 13.0 15.0 17.0
v 13.0 17.0 16.0
v 13.0 16.0 24.0
v 13.0 24.0 25.0
v 14.0 27.0 15.0
v 14.0 21.0 20.0
v 14.0 20.0 26.0
v 14.0 25.0 21.0
v 14.0 26.0 27.0
v 15.0 29.0 17.0
v 15.0 27.0 29.0
v 16.0 17.0 19.0
v 16.0 19.0 18.0
v 16.0 18.0 28.0
v 16.0 28.0 24.0
v 17.0 31.0 19.0
v 17.0 29.0 31.0
v 18.0 30.0 28.0
v 19.0 31.0 34.0
v 20.0 21.0 22.0
v 20.0 22.0 23.0
v 20.0 23.0 26.0
v 21.0 47.0 22.0
v 21.0 25.0 47.0
v 22.0 47.0 23.0
v 23.0 49.0 26.0
v 23.0 47.0 49.0
v 24.0 49.0 25.0
v 24.0 28.0 48.0
v 24.0 48.0 49.0
v 25.0 49.0 47.0
v 26.0 50.0 27.0
v 26.0 49.0 50.0
v 27.0 52.0 29.0
v 27.0 50.0 52.0
v 28.0 30.0 51.0
v 28.0 51.0 48.0
v 29.0 54.0 31.0
v 29.0 52.0 54.0
v 30.0 33.0 53.0
v 30.0 53.0 51.0
v 31.0 56.0 34.0
v 31.0 54.0 56.0
v 32.0 55.0 33.0
v 32.0 36.0 65.0
v 32.0 63.0 55.0
v 32.0 65.0 63.0
v 33.0 55.0 53.0
v 34.0 64.0 35.0
v 34.0 56.0 64.0
v 35.0 64.0 39.0
v 36.0 37.0 65.0
v 37.0 40.0 59.0
v 37.0 59.0 65.0
v 38.0 39.0 60.0
v 38.0 62.0 43.0
v 38.0 60.0 62.0
v 39.0 66.0 60.0
v 39.0 64.0 66.0
v 40.0 41.0 61.0
v 40.0 61.0 59.0
v 41.0 44.0 57.0
v 41.0 57.0 61.0
v 42.0 43.0 57.0
v 42.0 58.0 45.0
v 42.0 57.0 58.0
v 43.0 62.0 57.0
v 44.0 46.0 58.0
v 44.0 58.0 57.0
v 45.0 58.0 46.0
v 48.0 50.0 49.0
v 48.0 51.0 50.0
v 50.0 51.0 52.0
v 51.0 53.0 52.0
v 52.0 53.0 54.0
v 53.0 55.0 54.0
v 54.0 55.0 56.0
v 55.0 63.0 56.0
v 56.0 63.0 64.0
v 57.0 62.0 61.0
v 59.0 61.0 60.0
v 59.0 60.0 66.0
v 59.0 66.0 65.0
v 60.0 61.0 62.0
v 63.0 66.0 64.0
v 63.0 65.0 66.0

Mesh content:
v 23 16 29
v 23 16 33
v 23 19 29
v 23 19 33
v 24 19 30
v 24 19 32
v 24 21 30
v 24 21 32
v 26 21 31
v 26 58 31
v 28 16 30
v 28 16 32
v 29 1 30
v 29 1 31
v 29 1 32
v 29 3 30
v 29 3 32
v 29 4 30
v 29 4 32
v 30 0 31
v 31 0 30
v 31 0 31
v 31 0 32
v 31 1 29
v 31 1 30
v 31 1 32
v 31 1 33
v 31 3 29
v 31 3 33
v 31 4 29
v 31 4 33
v 31 16 27
v 31 16 28
v 31 16 34
v 31 16 35
v 31 19 27
v 31 19 28
v 31 19 34
v 31 19 35
v 31 21 28
v 31 21 29
v 31 21 33
v 31 21 34
v 31 58 29
v 31 58 33
v 31 63 31
v 32 0 31
v 33 1 30
v 33 1 31
v 33 1 32
v 33 3 30
v 33 3 32
v 33 4 30
v 33 4 32
v 34 16 30
v 34 16 32
v 36 21 31
v 36 58 31
v 38 19 30
v 38 19 32
v 38 21 30
v 38 21 32
v 39 16 29
v 39 16 33
v 39 19 29
v 39 19 33
f 1 2 3
f 1 11 2
f 1 3 36
f 1 33 11
f 1 32 33
f 1 36 32
f 2 4 3
f 2 35 4
f 2 11 12
f 2 12 35
f 3 4 6
f 3 6 5
f 3 5 36
f 4 38 6
f 4 35 39
f 4 39 38
f 5 6 8
f 5 8 7
f 5 7 37
f 5 37 36
f 6 43 8
f 6 38 43
f 7 8 9
f 7 9 40
f 7 40 37
f 8 42 9
f 8 43 42
f 9 10 41
f 9 45 10
f 9 41 40
f 9 42 45
f 10 44 41
f 10 46 44
f 10 45 46
f 11 18 12
f 11 30 18
f 11 33 30
f 12 18 19
f 12 19 34
f 12 34 35
f 13 14 15
f 13 25 14
f 13 15 17
f 13 17 16
f 13 16 24
f 13 24 25
f 14 27 15
f 14 21 20
f 14 20 26
f 14 25 21
f 14 26 27
f 15 29 17
f 15 27 29
f 16 17 19
f 16 19 18
f 16 18 28
f 16 28 24
f 17 31 19
f 17 29 31
f 18 30 28
f 19 31 34
f 20 21 22
f 20 22 23
f 20 23 26
f 21 47 22
f 21 25 47
f 22 47 23
f 23 49 26
f 23 47 49
f 24 49 25
f 24 28 48
f 24 48 49
f 25 49 47
f 26 50 27
f 26 49 50
f 27 52 29
f 27 50 52
f 28 30 51
f 28 51 48
f 29 54 31
f 29 52 54
f 30 33 53
f 30 53 51
f 31 56 34
f 31 54 56
f 32 55 33
f 32 36 65
f 32 63 55
f 32 65 63
f 33 55 53
f 34 64 35
f 34 56 64
f 35 64 39
f 36 37 65
f 37 40 59
f 37 59 65
f 38 39 60
f 38 62 43
f 38 60 62
f 39 66 60
f 39 64 66
f 40 41 61
f 40 61 59
f 41 44 57
f 41 57 61
f 42 43 57
f 42 58 45
f 42 57 58
f 43 62 57
f 44 46 58
f 44 58 57
f 45 58 46
f 48 50 49
f 48 51 50
f 50 51 52
f 51 53 52
f 52 53 54
f 53 55 54
f 54 55 56
f 55 63 56
f 56 63 64
f 57 62 61
f 59 61 60
f 59 60 66
f 59 66 65
f 60 61 62
f 63 66 64
f 63 65 66

Number of vertices: 66
Number of faces: 128
Mesh saved to: sword.glb
```

## References
- [LLaMa-Mesh](https://github.com/nv-tlabs/LLaMA-Mesh)
- [meshgen](https://github.com/huggingface/meshgen)
