Following https://huggingface.co/blog/annotated-diffusion with slight changes.

## Setup
```
conda env create -n diffusion -f environment.yml
conda activate diffusion
```
You may need to install PyTorch manually for CUDA support. See https://pytorch.org/get-started/locally/ for more details.

## Relevant files
- `run.ipynb`: Testing interface
- `forward.py`: Forward diffusion process
- `backward.py`: Backward diffusion process
- `nn.py`: PyTorch modules
- `data.py`: Data loader
- `utils.py`: Various utilities

## Style
Loosely adhering to [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and formatting with [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)