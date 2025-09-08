import json, torch
from main import load_model
from sentence_transformers import SentenceTransformer
from basis import tao_construction
from vis import plot_matrices, plot_lines, plot_matrix

# with open("/Users/mrmackamoo/Projects/mechanistic-interpretability/understanding_superposition/data/mpnet2_words.json", "r") as f:
#     all_words = json.load(f)

basis = tao_construction(0, 50, width=225)

mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embedding_module = mpnet[0].auto_model.embeddings.word_embeddings

words = ["apple", "banana", "red", "human",]

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

codes = model.encoder(embeddings)

basis = torch.tensor(basis, device=codes.device, dtype=codes.dtype)

def axis_knn(codes, axes: list[int] = [], k=5):
    from sklearn.neighbors import NearestNeighbors
    if axes:
        codes_np = codes[:, axes].detach().cpu().numpy()
    else:
        codes_np = codes.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(codes_np)
    distances, indices = nbrs.kneighbors(codes_np)
    return indices[:, 1:]  # exclude self

def decision_tree(codes, labels):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    codes_np = codes.detach().cpu().numpy()
    X_train, X_test, y_train, y_test = train_test_split(codes_np, labels, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")

    return clf