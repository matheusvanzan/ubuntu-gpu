# ubuntu-gpu

## NVIDIA Drivers

### Install

Before installing the Nvidia driver, ensure that the **driver version** is **compatible with** the **CUDA Toolkit** you intend to install.

After that, you can proceed to the [driver download page](https://www.nvidia.com/download/index.aspx), where you should specify your machine’s specifications.

### Check installed version

```bash
$ nvidia-smi
```

## Sample Code

### Using PyTorch

```python
import torch

if torch.cuda.is_available():
    print(f"Available GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available.")
```
