# Environment Setup (no venv required):
- Follow the instructions to install uv here, https://docs.astral.sh/uv/getting-started/installation/
- Hit a cheeky `git clone`
- `cd` into the cloned directory
- Follow below

# Pytorch Installation for System Without Discrete GPU:

``` python-console
uv sync --extra cpu
```

# Pytorch Installation for Nvidia:

``` python-console
uv sync --extra cuda
```
