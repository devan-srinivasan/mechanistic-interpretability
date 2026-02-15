from sae import SAE
import torch, torch.nn.functional as F
from tqdm import tqdm

model = SAE(embed_dim=768, hidden_dim=768 * 256)
model.load_state_dict(torch.load("understanding_superposition/runs/moonlit-heartthrob-128_20260215_133409/model_epoch_20.pth"))

val_datafile = "understanding_superposition/data/bert_words.pt/val_0.pt"
val_tensor = torch.load(val_datafile)

reconstructed, codes = model(val_tensor)

correct = 0
for i, tensor in tqdm(enumerate(val_tensor)):
    similarities = F.cosine_similarity(reconstructed, tensor.unsqueeze(0), dim=1)
    best_match_idx = torch.argmax(similarities).item()
    if best_match_idx == i:
        correct += 1
print(f"{correct}/{len(val_tensor)} correct")