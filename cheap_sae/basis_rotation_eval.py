import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig, BertForMaskedLM
from datasets import load_dataset
from helpers import token_batches, _run_dev_eval, UVRotation
import json

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

cfg = BertConfig.from_pretrained("bert-base-cased")
cfg._attn_implementation = "eager"

model = BertForMaskedLM.from_pretrained("bert-base-cased", config=cfg)
model.eval()  # we'll freeze BERT

layer = 6
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
    
# Load a previously-saved UVRotation (U, V) and construct the module from it
ckpt_path = os.path.join("cheap_sae/artifacts/", f"bert_qproj_layer{layer}_uv_rotation.pt")
save_obj = torch.load(ckpt_path, map_location="cpu")

uv = UVRotation(d=model.config.hidden_size).to(device)
with torch.no_grad():
    uv.U.copy_(save_obj["U"].to(uv.U.device, dtype=uv.U.dtype))
    uv.V.copy_(save_obj["V"].to(uv.V.device, dtype=uv.V.dtype))

uv.eval()

activation_indices = []

def _eval_hook(module, input, output):
    global activation_indices
    X = input[0].detach()
    z_prime, z_orig, sparse_term, UW = uv(X, W, b)

    active_mask = sparse_term.abs() > 0.2  # (B, N, d)

    B, N, d = sparse_term.shape
    K = 50

    abs_vals = sparse_term.abs()
    masked_abs = abs_vals.masked_fill(~active_mask, float("-inf"))  # ignore inactive

    topk_vals, topk_idx = torch.topk(masked_abs, k=K, dim=-1, largest=True, sorted=False)  # (B, N, K)
    topk_raw_vals = sparse_term.gather(-1, topk_idx)  # signed values, (B, N, K)

    # K is an upper bound: keep only truly-active entries, pad the rest.
    # Convert the (possibly -inf) masked top-k scores into a validity mask.
    valid = torch.isfinite(topk_vals)  # (B, N, K)

    # Move valid entries to the front (stable w.r.t. current topk order), pad the rest.
    order = valid.to(torch.int64).argsort(dim=-1, descending=True)  # valid first
    topk_idx = topk_idx.gather(-1, order)
    topk_raw_vals = topk_raw_vals.gather(-1, order)

    # Pad invalid tail with (-1, 0.0)
    valid_sorted = valid.gather(-1, order)
    topk_idx = topk_idx.masked_fill(~valid_sorted, -1)
    topk_raw_vals = topk_raw_vals.masked_fill(~valid_sorted, 0.0)

    feature_indices = torch.stack(
        (topk_idx.to(topk_raw_vals.dtype), topk_raw_vals),
        dim=-1,
    )  # (B, N, K, 2)

    activation_indices.extend(feature_indices)

    return z_prime.to(output.device)

uv.eval()
model.eval()

# base_total_loss, n_base_masked = _run_dev_eval(model, ds["validation"], batch_size, tokenizer, device, max_length)
eval_handle = model.bert.encoder.layer[layer].attention.self.query.register_forward_hook(_eval_hook)
new_total_loss, n_new_masked = _run_dev_eval(model, ds["validation"], batch_size, tokenizer, device, max_length)
eval_handle.remove()

# activation_indices is a list of tensors shaped (B, 128, K, 2); concatenate across batches (dim=0)
activation_indices_cat = torch.cat(
    [t.unsqueeze(0).detach().cpu() for t in activation_indices],  # ensure they're all on the same device/dtype
    dim=0
)  # -> (sum_B, 128, K, 2)
print(activation_indices_cat.shape)

save_dir = os.path.join("cheap_sae", "features", f"layer{layer}_qrot")
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

# print(f"Baseline MLM loss: {(base_total_loss / n_base_masked):.4f}")
print(f"MLM loss with UV rotation: {(new_total_loss / n_new_masked):.4f}")