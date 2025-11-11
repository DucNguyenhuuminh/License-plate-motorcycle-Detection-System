import torch

# 1. Kiểm tra CUDA có sẵn không
print(f"CUDA Available: {torch.cuda.is_available()}")

# 2. Kiểm tra phiên bản CUDA PyTorch đang sử dụng
print(f"PyTorch CUDA Version: {torch.version.cuda}")