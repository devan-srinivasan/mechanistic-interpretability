import json, torch
from main import load_model
from sentence_transformers import SentenceTransformer
from basis import tao_construction
from vis import plot_matrices, plot_lines, plot_matrix

with open("/Users/mrmackamoo/Projects/mechanistic-interpretability/understanding_superposition/data/mpnet2_words.json", "r") as f:
    dictionary = json.load(f)

basis = tao_construction(0, 50, width=225)

mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embedding_module = mpnet[0].auto_model.embeddings.word_embeddings

words = ["happy", "joy", "cheer", "sad", "depression", "melancholy"]

with torch.no_grad():
    # may need to batch this if too large
    tokens = torch.tensor([e[1] for e in mpnet[0].tokenizer(words)['input_ids']])
    embeddings = embedding_module(tokens.to(mpnet.device))
    embeddings = embeddings.to(torch.float32)

model = load_model(
    checkpoint="mrmackamoo/mechanistic-interpretability/model:v12", 
    tmp_dir="/Users/mrmackamoo/Projects/mechanistic-interpretability/understanding_superposition/tmp",
    download=False
)

model.to(embeddings.device)

reconstructed = model(embeddings)
codes = model.encoder(embeddings)

basis = torch.tensor(basis, device=codes.device, dtype=codes.dtype)

print(codes.shape)

cosine_sim_matrix = torch.nn.functional.cosine_similarity(
    codes.unsqueeze(1), codes.unsqueeze(0), dim=-1
)

codes_scaled = 2 * (codes - codes.min()) / (codes.max() - codes.min()) - 1
codes_scaled = codes_scaled.detach().to("cpu")

print(codes_scaled[:, 185:190])

plot_matrices([codes_scaled[:, i: i+15] for i in range(0, codes.shape[1], 15)])