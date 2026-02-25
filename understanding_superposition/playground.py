# from datasets import load_dataset
# import torch

# # Load the WikiText dataset with streaming enabled
# # dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

# encoder = torch.nn.Linear(768, 64)

# rand = torch.ones((5, 512, 768))

# print(encoder(rand).shape)

from sae import SAE
import torch

dev = "cuda:8"

input = torch.randn((64, 512, 768)).to(dev)

model = SAE(embed_dim=768, hidden_dim=768 * 50).to(dev)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

reconstructed, codes = model(input)

loss = torch.nn.functional.mse_loss(reconstructed, input)

loss.backward()

optim.step()

