import torch
print(f"PyTorch version: {torch.__version__}")

x = torch.rand(5, 3)
print(x)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")