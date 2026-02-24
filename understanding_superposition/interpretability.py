from sae import SAE
import torch, torch.nn.functional as F
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

print("Loading model & data...")
model = SAE(embed_dim=768, hidden_dim=768 * 256)
model.load_state_dict(torch.load("understanding_superposition/runs/glowing-jazz-155_20260217_201128/model_epoch_2.pth/model.pth", map_location="cpu"))

val_datafile = "understanding_superposition/data/bert_words/val_0.pt"
val_tensor = torch.load(val_datafile)

with open("understanding_superposition/data/bert_words/val.txt", "r") as f:
    words = [line.strip() for line in f]

print("Running model...")
reconstructed, codes = model(val_tensor)

# shape (N, n_f)
codes_active = (codes > 0.01).float()

# shape (n_f,)
activity = codes_active.sum(dim=0)

# shape (n_f,)
active_feature_mask = activity > 0

# shape (N,)
modelled_words_mask = codes_active.sum(dim=-1) > 0

codes_filtered = codes[modelled_words_mask][:, active_feature_mask]

words_ = [w for i,w in enumerate(words) if modelled_words_mask[i]]

# print(f"mean_cos_sim: {F.cosine_similarity(reconstructed[modelled_words_mask], val_tensor[modelled_words_mask]).mean().item():.4f}")

def get_words_fired(feature_idx):
    return [words_[i] for i in torch.where(codes_filtered[:, feature_idx] > 0.001)[0]]

# shape (n_f,)
fire_counts = (codes_filtered > 0).sum(dim=0)

# plt.hist(fire_counts.numpy(), bins=range(fire_counts.min().item(), fire_counts.max().item() + 2), edgecolor='black')
# plt.xlabel("Fire Count")
# plt.ylabel("Frequency")
# plt.title("Histogram")
# plt.show()

fire_indices_sorted_ = torch.argsort(fire_counts)

s = 1200
for idx in fire_indices_sorted_[s:s+20]:
    print(f"{idx.item()}: {get_words_fired(idx)}")

print()