from sae import SAE
import torch, torch.nn.functional as F
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

model = SAE(embed_dim=768, hidden_dim=768 * 256)
model.load_state_dict(torch.load("understanding_superposition/tmp/model_epoch_107.pth/model.pth", map_location="cpu"))

val_datafile = "understanding_superposition/data/bert_words/val_0.pt"
val_tensor = torch.load(val_datafile)

with open("understanding_superposition/data/bert_words.json", "r") as f:
    vocab = json.load(f)

with open("understanding_superposition/data/bert_words/indices.txt", "r") as f:
    indices = [int(line.strip()) for line in f]

num_samples = len(indices)
train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)

val_indices = indices[train_size:train_size + val_size]

val_words = [vocab[idx] for idx in val_indices]

reconstructed, codes = model(val_tensor)

codes_active = (codes > 0.005).float()

activity = codes_active.sum(dim=0)

# c = 0
# for i, v in tqdm(enumerate(val_tensor), leave=False):
#     cos_sim = F.cosine_similarity(reconstructed, val_tensor[i].unsqueeze(0))
#     if cos_sim.argmax() == i:
#         c += 1

# print(f"cos acc %: {c / len(val_tensor) * 100:.2f}%")

cosine_sims = F.cosine_similarity(reconstructed, val_tensor)
print(f"cos_error: {(1 - cosine_sims).mean().item():.4f}")

print(f"n_active: {(activity > 0).sum().item()}")
print(f"mse: {F.mse_loss(reconstructed, val_tensor).item()}")

# histogram = torch.zeros(40, dtype=torch.int32)
# for value in activity.int():
#     histogram[value.item() - 1] += 1

# # Plot the histogram
# plt.bar(range(1, 41), histogram.numpy())
# plt.xlabel('n_words_fired_in_val')
# plt.ylabel('n_features')
# plt.title('Feature Density')
# plt.show()  

