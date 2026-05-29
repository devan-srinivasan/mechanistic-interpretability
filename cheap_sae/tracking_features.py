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

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# clean ds a bit
ds = ds.filter(lambda x: len(x["text"].split()) > 5 and not x["text"].startswith(" = "))

batch_size = 32

master_dir = "/Users/mrmackamoo/Projects/topological_analysis/manifold_evolution/results/feature_tracking/bert/wikitext-103/"
os.makedirs(master_dir, exist_ok=True)

t = {}

def select_significant_features(tensor):
    ...

def attention_hook(module, input, output, layer):
    global t
    X = input[0]
    Q, K = module.self.query(X), module.self.key(X)

    t[f'layer[{layer}].attn.Q'] = ...
    t[f'layer[{layer}].attn.K'] = ...

    W_v, b_v = module.self.value.weight, module.self.value.bias
    W_o, b_o = module.output.dense.weight, module.output.dense.bias
    
    W_OV_w = W_o @ W_v
    W_OV_b = W_o @ b_v + b_o
    
    OV = X @ W_OV_w.T + W_OV_b

    t[f'layer[{layer}].attn.OV'] = ...

    return output

def mlp_act_hook(module, input, output, layer):
    global t
    t[f'layer[{layer}].mlp.act'] = ...
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

    # Save results in the directory
    with open(os.path.join(batch_dir, "tokens.txt"), "w") as f:
        for token_list in tokens:
            f.write(" ".join(token_list) + "\n")
    
    torch.save(...)
    torch.save(inputs["attention_mask"], os.path.join(batch_dir, "attention_mask.pt"))

for h in handles:
    h.remove()