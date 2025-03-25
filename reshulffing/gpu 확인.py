import torch

print("CUDA Available:", torch.cuda.is_available())
print("PyTorch Version:", torch.__version__)
print("Current Device:", torch.cuda.current_device())
print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Allocated Memory:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("Cached Memory:", torch.cuda.memory_reserved() / 1024**2, "MB")
