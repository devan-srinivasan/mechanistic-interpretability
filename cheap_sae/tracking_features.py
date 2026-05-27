import torch, numpy as np
from transformers import BertModel, BertTokenizer, BertConfig
from tqdm import tqdm
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from datasets import load_dataset
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

cfg = BertConfig.from_pretrained("bert-base-cased")
cfg._attn_implementation = "eager"

model = BertModel.from_pretrained("bert-base-cased", config=cfg)

def serialize_to_matrix(t):
    Q = np.stack([
        t[f'layer[{layer}].attn.Q'] for layer in range(12)
    ], axis=0)
    K = np.stack([
        t[f'layer[{layer}].attn.K'] for layer in range(12)
    ], axis=0)
    OV = np.stack([
        t[f'layer[{layer}].attn.OV'] for layer in range(12)
    ], axis=0)
    MLP = np.stack([
        t[f'layer[{layer}].mlp.act'] for layer in range(12)
    ], axis=0)
    MLP = MLP.reshape(12 * 4, MLP.shape[1], MLP.shape[2], MLP.shape[3] // 4)

    return np.concatenate([Q, K, OV, MLP], axis=0).transpose(1, 0, 2, 3)

def decode_matrix(M):
    Q = M[:, :12]
    K = M[:, 12:24]
    OV = M[:, 24:36]
    MLP = M[:, 36:].reshape(M.shape[0], 12, M.shape[2], 4 * 768)

    return Q, K, OV, MLP

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# clean ds a bit
ds = ds.filter(lambda x: len(x["text"].split()) > 5 and not x["text"].startswith(" = "))

batch_size = 32

master_dir = "/Users/mrmackamoo/Projects/topological_analysis/manifold_evolution/results/feature_tracking/bert/wikitext-103/"
os.makedirs(master_dir, exist_ok=True)

t = {}

def attention_hook(module, input, output, layer):
    global t
    X = input[0]
    Q, K = module.self.query(X), module.self.key(X)

    t[f'layer[{layer}].attn.Q'] = Q.detach().numpy()
    t[f'layer[{layer}].attn.K'] = K.detach().numpy()

    W_v, b_v = module.self.value.weight, module.self.value.bias
    W_o, b_o = module.output.dense.weight, module.output.dense.bias
    
    W_OV_w = W_o @ W_v
    W_OV_b = W_o @ b_v + b_o
    
    OV = X @ W_OV_w.T + W_OV_b

    t[f'layer[{layer}].attn.OV'] = OV.detach().numpy()

    return output

def mlp_act_hook(module, input, output, layer):
    global t
    t[f'layer[{layer}].mlp.act'] = output.detach().numpy()
    return output

handles = []

for layer in range(12):

    handles.extend([
        model.encoder.layer[layer].attention.register_forward_hook(lambda m, i, o, layer=layer: attention_hook(m, i, o, layer)),
        model.encoder.layer[layer].intermediate.intermediate_act_fn.register_forward_hook(lambda m, i, o, layer=layer: mlp_act_hook(m, i, o, layer)),
    ])

n_samples = 5000 # len(ds["train"])
for batch_idx in tqdm(range(0, n_samples, batch_size)):
    batch = ds["train"].select(range(batch_idx, min(batch_idx + batch_size, n_samples)))
    texts = [sample["text"] for sample in batch]
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    batch_dir = os.path.join(master_dir, f"batch_{batch_idx // batch_size:03d}")
    os.makedirs(batch_dir, exist_ok=True)

    with torch.no_grad():
        _ = model(**inputs)

    tokens = [tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]) for i in range(inputs['input_ids'].shape[0])]

    M = serialize_to_matrix(t)

    # Save results in the directory
    with open(os.path.join(batch_dir, "tokens.txt"), "w") as f:
        for token_list in tokens:
            f.write(" ".join(token_list) + "\n")
    
    torch.save(M, os.path.join(batch_dir, "activations.pt"))
    torch.save(inputs["attention_mask"], os.path.join(batch_dir, "attention_mask.pt"))

for h in handles:
    h.remove()