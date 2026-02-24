from datasets import load_dataset
import torch

# Load the WikiText dataset with streaming enabled
# dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

encoder = torch.nn.Linear(768, 64)

rand = torch.ones((5, 512, 768))

print(encoder(rand).shape)