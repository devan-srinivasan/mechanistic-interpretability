import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

def print_memory_usage(dev):
    total_memory = torch.cuda.get_device_properties(dev).total_memory / 1e9
    allocated_memory = torch.cuda.memory_allocated(dev) / 1e9
    reserved_memory = torch.cuda.memory_reserved(dev) / 1e9
    free_memory = reserved_memory - allocated_memory
    print(f"Allocated: {allocated_memory:.2f} GB, Reserved but Unallocated: {free_memory:.2f} GB, Total: {total_memory:.2f} GB")

dev = "cuda:7"

print("Usage before input:")
print_memory_usage(dev)

input = torch.randn((64, 512, 768)).to(dev)

print("Usage before model:")
print_memory_usage(dev)

from sae import SAE

model = SAE(embed_dim=768, hidden_dim=768 * 50).to(dev)

print("Usage before forward pass:")
print_memory_usage(dev)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

reconstructed, codes = model(input)

print("Usage before backward pass:")
print_memory_usage(dev)

loss = torch.nn.functional.mse_loss(reconstructed, input)

loss.backward()

optim.step()

print("Usage after everything:")
print_memory_usage(dev)
