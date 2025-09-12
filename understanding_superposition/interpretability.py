import json, torch, os
from main import load_model
from sentence_transformers import SentenceTransformer
from basis import tao_construction
from vis import plot_matrices, plot_lines, plot_matrix, plot_clusters
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import wordnet
from tqdm import tqdm

with open("/Users/mrmackamoo/Projects/mechanistic-interpretability/understanding_superposition/data/mpnet2_words.json", "r") as f:
    all_words = json.load(f)

basis = tao_construction(0, 50, width=225)

all_words = ["runner", "player", "walker", "singer", "run", "play", "walk", "sing",
             "jumper", "jump", "dancer", "dance"]

mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embedding_module = mpnet[0].auto_model.embeddings.word_embeddings

with torch.no_grad():
    # may need to batch this if too large
    tokens = torch.tensor([e[1] for e in mpnet[0].tokenizer(all_words)['input_ids']])
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
        # print_tree(0)
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

pos_dict = get_pos_dictionary(all_words, "/Users/mrmackamoo/Projects/mechanistic-interpretability/understanding_superposition/data/mpnet2_word_pos.json")
non_empty_pos = {word: tags for word, tags in pos_dict.items() if tags}
print(f"{len(non_empty_pos)}/{len(pos_dict)} words have non-empty POS lists ({len(non_empty_pos)/len(pos_dict):.2%})")

