import torch

print("GPU available:", torch.cuda.is_available())
import torch

print(torch.version.cuda)  # should print CUDA version, e.g., 11.8
print(torch.cuda.is_available())  # should print True
