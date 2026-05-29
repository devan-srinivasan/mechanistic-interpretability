import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from datasets import load_dataset

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

cfg = BertConfig.from_pretrained("bert-base-cased")
cfg._attn_implementation = "eager"
model = BertModel.from_pretrained("bert-base-cased", config=cfg)
model.eval()  # we'll freeze BERT

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# clean ds a bit
ds = ds.filter(lambda x: len(x["text"].split()) > 5 and not x["text"].startswith(" = "))

layer = 6
bert_layer = model.encoder.layer[layer]
module = bert_layer.attention.self.query 

acts = {} # this will constantly be overwritten
def _hook(module, input, output):
    global acts
    # module_in is a tuple; for nn.Linear in BERT it's (X,) where X is [B, T, d]
    acts["x_in"] = input[0].detach()
    acts["y_out"] = output.detach()

handle = module.register_forward_hook(_hook)

sample = ds["train"][10000]["text"]
inputs = tokenizer(sample, return_tensors="pt")
with torch.no_grad():
    model(**inputs)

X = acts["x_in"][0]
Y = acts["y_out"][0]
W = module.weight.detach()
b = module.bias.detach()

# file_path
fp = "cheap_sae/artifacts/bert_qproj_layer6_uv_rotation.pt"

obj = torch.load(fp)

U, V = obj["U"], obj["V"]

UW = U @ W
Vx = X @ V.T
Y_prime = Vx @ UW.T + b @ U.T
sparse_term = X @ UW.T
print()
