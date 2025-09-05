import json, torch
from main import load_model
from sentence_transformers import SentenceTransformer

mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embedding_module = mpnet[0].auto_model.embeddings.word_embeddings

words = ["apple", "banana", "chair", "table", "red", "yellow"]

with torch.no_grad():
    # may need to batch this if too large
    tokens = torch.tensor([e[1] for e in mpnet[0].tokenizer(words)['input_ids']])
    embeddings = embedding_module(tokens.to(mpnet.device))
    embeddings = embeddings.to(torch.float32)


model = load_model(
    checkpoint="mrmackamoo/mechanistic-interpretability/model:v6", 
    tmp_dir="/Users/mrmackamoo/Projects/mechanistic-interpretability/understanding_superposition/tmp",
    download=False
)

model.to(embeddings.device)

reconstructed = model(embeddings)

recon_sim = torch.nn.functional.cosine_similarity(
    reconstructed.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
)

true_sim = torch.nn.functional.cosine_similarity(
    embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
)

recon_sorted_indices = torch.argsort(recon_sim, dim=1, descending=True)
true_sorted_indices = torch.argsort(true_sim, dim=1, descending=True)

k = 3

top_k_recon = recon_sorted_indices[:, :k]
top_k_true = true_sorted_indices[:, :k]

# compute how much overlap in topk there is row-wise
overlap = (top_k_recon.unsqueeze(2) == top_k_true.unsqueeze(1)).any(dim=2).float().mean(dim=1)


print(reconstructed)