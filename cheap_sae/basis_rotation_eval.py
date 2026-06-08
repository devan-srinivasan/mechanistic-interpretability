import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig, BertForMaskedLM
from datasets import load_dataset
from helpers import token_batches, _run_dev_eval, Transformation
import json

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

cfg = BertConfig.from_pretrained("bert-base-cased")
cfg._attn_implementation = "eager"

model = BertForMaskedLM.from_pretrained("bert-base-cased", config=cfg)
model.eval()  # we'll freeze BERT

layer = 11
max_length = 128
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

module = model.bert.encoder.layer[layer].attention.self.query  # nn.Linear(hidden_size, all_head_size)
W = module.weight.detach()                # [out_dim, in_dim] = [hidden, hidden] for BERT
b = module.bias.detach() 

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
ds = ds.filter(lambda x: len(x["text"].split()) > 5 and not x["text"].startswith(" = "))
if torch.mps.is_available():
    ds['train'] = ds['train'].shuffle(seed=42).select(range(64))
    # ds['validation'] = ds['validation'].shuffle(seed=42).select(range(500))

save_obj = torch.load(f"cheap_sae/artifacts/bert_qproj_layer{layer}_transformation.pt", map_location="cpu")

# load the model here
transformation = Transformation(d=W.shape[0], init="rand").to(device)
with torch.no_grad():
    transformation.T.copy_(save_obj["T"].to(device))
    transformation.T_.copy_(save_obj["T_"].to(device))

activation_indices = []

def _eval_hook(module, input, output):
    global activation_indices
    X = input[0].detach()

    z_prime, z_orig, sparse_term = transformation(X, W.to(device), b.to(device))

    activation_indices.extend(sparse_term.detach().cpu())

    return z_prime.to(output.device)

transformation.eval()
model.eval()

base_total_loss, n_base_masked = _run_dev_eval(model, ds["validation"], batch_size, tokenizer, device, max_length)

print(f"Baseline MLM loss: {(base_total_loss / n_base_masked):.4f}")

eval_handle = model.bert.encoder.layer[layer].attention.self.query.register_forward_hook(_eval_hook)
new_total_loss, n_new_masked = _run_dev_eval(model, ds["validation"], batch_size, tokenizer, device, max_length)
eval_handle.remove()
print(f"MLM loss with Transformation: {(new_total_loss / n_new_masked):.4f}")

# activation_indices is a list of tensors shaped (B, 128, K, 2); concatenate across batches (dim=0)
# activation_indices_cat = torch.cat(
#     [t.unsqueeze(0).detach().cpu() for t in activation_indices],  # ensure they're all on the same device/dtype
#     dim=0
# )  # -> (sum_B, 128, K, 2)

activation_indices_cat = torch.cat(
    [t.unsqueeze(0).detach().cpu() for t in activation_indices],  # ensure they're all on the same device/dtype
    dim=0
)  # -> (sum_B, 128, d)
print(activation_indices_cat.shape)

save_dir = os.path.join("cheap_sae", "features", f"layer{layer}_T")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "activations.pt")
torch.save(activation_indices_cat, save_path)
print(f"Saved activations to: {save_path}")

all_tokens = []
for batch in token_batches(ds["validation"], batch_size, tokenizer, device, max_length):
    # batch["input_ids"]: (B, N)
    input_ids = batch["input_ids"].detach().cpu().tolist()
    batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
    all_tokens.extend(batch_tokens)

tokens_path = os.path.join(save_dir, "tokens.json")
with open(tokens_path, "w", encoding="utf-8") as f:
    json.dump(all_tokens, f, ensure_ascii=False)

print(f"Saved tokens to: {tokens_path}")