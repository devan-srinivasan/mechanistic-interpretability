import json, torch, os, random, numpy as np
from tao_manifold_learning import load_model
from sentence_transformers import SentenceTransformer
from basis import tao_construction
from vis import plot_matrices, plot_lines, plot_matrix, plot_clusters
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import wordnet
from tqdm import tqdm
from sklearn.metrics import mutual_info_score


with open("/Users/mrmackamoo/Projects/mechanistic-interpretability/understanding_superposition/data/mpnet2_words.json", "r") as f:
    all_words = json.load(f)

basis = tao_construction(0, 760, width=784)

mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embedding_module = mpnet[0].auto_model.embeddings.word_embeddings

with torch.no_grad():
    # may need to batch this if too large
    tokens = torch.tensor([e[1] for e in mpnet[0].tokenizer(all_words)['input_ids']])
    embeddings = embedding_module(tokens.to(mpnet.device))
    embeddings = embeddings.to(torch.float32)

model = load_model(
    checkpoint="mrmackamoo/mechanistic-interpretability/model:v14", 
    tmp_dir="/Users/mrmackamoo/Projects/mechanistic-interpretability/understanding_superposition/tmp",
    download=False
)

model.to(embeddings.device)

codes = model.encoder(embeddings)

basis = torch.tensor(basis, device=codes.device, dtype=codes.dtype)[:, :codes.shape[1]]

def axes_knn(codes, axes: list[int] = [], k=5):
    from sklearn.neighbors import NearestNeighbors
    if axes:
        codes_np = codes[:, axes].detach().cpu().numpy()
    else:
        codes_np = codes.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(codes_np)
    distances, indices = nbrs.kneighbors(codes_np)
    return indices[:, 1:]  # exclude self

def decision_tree(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor):
    tree = DecisionTreeClassifier()
    tree.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

    def print_tree(node, depth=0):
        tree_ = tree.tree_
        indent = '\t' * depth
        if tree_.feature[node] != -2:  # not a leaf
            print(f"{indent}Node {node}: feature_idx={tree_.feature[node]}, threshold={tree_.threshold[node]:.4f}")
            print_tree(tree_.children_left[node], depth + 1)
            print_tree(tree_.children_right[node], depth + 1)
        else:
            print(f"{indent}Leaf {node}: value={tree_.value[node]}")

    predictions = tree.predict(X_test.detach().cpu().numpy())
    
    if len(y_test) < 10:
        print_tree(0)
        print("Predicted classes:", predictions)
    else:
        accuracy = (predictions == y_test.detach().cpu().numpy()).mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")

def axes_k_means(codes, axes: list[int] = [], n_clusters=5):
    from sklearn.cluster import KMeans
    if axes:
        codes_np = codes[:, axes].detach().cpu().numpy()
    else:
        codes_np = codes.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0, algorithm="elkan").fit(codes_np)
    labels = kmeans.labels_.tolist()
    return labels  # list of integers, length = codes.shape[0]

def get_pos_dictionary(words: list[str], pos_tags_file: str) -> dict[str, list[str]]:
    """
    for each word in words, it gets all possible part-of-speech tags using WordNet (considering all applicable synsets)
    returns a dictionary mapping each word to its list of POS tags (which could be empty if no synsets found)
    """

    if os.path.exists(pos_tags_file):
        with open(pos_tags_file, "r") as f:
            return json.load(f)
    pos_dict = {}
    for word in tqdm(words, unit='word'):
        synsets = wordnet.synsets(word)
        pos_tags = set(synset.pos() for synset in synsets)
        pos_dict[word] = list(pos_tags)
    with open(pos_tags_file, "w") as f:
        json.dump(pos_dict, f)
    return pos_dict

def create_pos_data(train_ratio=0.85) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Creates train & test splits for pos classification based on the POS tags from WordNet.
    returns X_train, y_train, X_test, y_test, filtered_words (words with non-empty POS tags)
    """
    pos_dict = get_pos_dictionary(all_words, "/Users/mrmackamoo/Projects/mechanistic-interpretability/understanding_superposition/data/mpnet2_word_pos.json")
    non_empty_pos = {word: tags for word, tags in pos_dict.items() if tags}
    print(f"{len(non_empty_pos)}/{len(pos_dict)} words have non-empty POS lists ({len(non_empty_pos)/len(pos_dict):.2%})")

    all_pos_tags = set(tags[0] for tags in pos_dict.values() if tags)

    mask = [bool(pos_dict[word]) for word in all_words]
    filtered_words = [word for word, has_pos in zip(all_words, mask) if has_pos]
    filtered_codes = codes[torch.tensor(mask, device=codes.device)]

    pos_to_indices = {pos: [] for pos in all_pos_tags}
    for idx, word in enumerate(filtered_words):
        pos = pos_dict[word][0]
        pos_to_indices[pos].append(idx)

    pos_to_int = {pos: i for i, pos in enumerate(all_pos_tags)}
    train_indices, test_indices = [], []
    y_train, y_test = [], []

    for pos, indices in pos_to_indices.items():
        indices_copy = indices.copy()
        random.shuffle(indices_copy)
        split_idx = int(train_ratio * len(indices_copy))
        train_indices.extend(indices_copy[:split_idx])
        y_train.extend([pos_to_int[pos]] * split_idx)
        test_indices.extend(indices_copy[split_idx:])
        y_test.extend([pos_to_int[pos]] * (len(indices_copy) - split_idx))

    y_train = torch.tensor(y_train, device=filtered_codes.device)
    y_test = torch.tensor(y_test, device=filtered_codes.device)
    X_train = filtered_codes[torch.tensor(train_indices, device=filtered_codes.device)]
    X_test = filtered_codes[torch.tensor(test_indices, device=filtered_codes.device)]

    return X_train, y_train, X_test, y_test, filtered_words

def get_nn(word, emb_matrix=embeddings, most=True):
    """
    Returns the top 5 most (or least, if most=False) similar words to the given word,
    along with their cosine similarity scores.
    """
    import torch.nn.functional as F

    if word not in all_words:
        raise ValueError(f"Word '{word}' not found in all_words.")

    idx = all_words.index(word)
    query_vec = emb_matrix[idx]
    emb_flat = emb_matrix
    # Compute cosine similarity in a vectorized way
    sims = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), emb_flat, dim=-1)
    
    # Exclude self
    sims[idx] = -float('inf') if most else float('inf')
    if most:
        topk = torch.topk(sims, 5)
    else:
        topk = torch.topk(-sims, 5)
        values, indices = topk.values * -1, topk.indices
    results = []
    if most:
        for sim, i in zip(topk.values.tolist(), topk.indices.tolist()):
            results.append((all_words[i], f"{sim:.3f}"))
    else:
        for sim, i in zip(values.tolist(), indices.tolist()):
            results.append((all_words[i], f"{sim:.3f}"))
    return results

feature_idx = 0
highly_active_indices = torch.where(torch.abs(codes[:, feature_idx]) > 1e-2)[0].tolist()

for idx in highly_active_indices:
    print(all_words[idx])