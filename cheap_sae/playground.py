import torch, json, numpy as np, nltk, torch.nn.functional as F, os
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from helpers import Transformation, _run_dev_eval, SAE, MLPSAE
from datasets import load_dataset
from torch.nn.functional import cosine_similarity

def load_model(path):
    model_save = torch.load(path, map_location="cpu")
    U, S = model_save['U'], model_save['S']
    transformation = SAE(d=U.shape[0], init="rand")
    transformation.U.data.copy_(U)
    transformation.S.data.copy_(S)
    transformation.eval()
    return transformation

def load_mlp_model(path):
    model_save = torch.load(path, map_location="cpu")
    U, S = model_save['U'], model_save['S']
    transformation = MLPSAE(d1=U.shape[0], d2=S.shape[0], init="rand")
    transformation.U.data.copy_(U)
    transformation.S.data.copy_(S)
    transformation.eval()
    return transformation

q_t = load_model("cheap_sae/artifacts/sae_layer11_correct/sae_q_layer11_sparse1.0_inv1.0_rel_match1.0.pt")
k_t = load_model("cheap_sae/artifacts/sae_layer11_correct/sae_k_layer11_sparse1.0_inv1.0_rel_match1.0.pt")
# v_t = load_model("cheap_sae/artifacts/sae_layer11_correct/sae_v_layer11_sparse1.0_inv1.0_rel_match1.0.pt")
o_t = load_model("cheap_sae/artifacts/sae_layer11_correct/sae_o_layer11_sparse1.0_inv1.0_rel_match1.0.pt")

mlp_t = load_mlp_model("cheap_sae/artifacts/sae_layer11_correct/mlp_sae_mlp_layer11_sparse1.0_rel_match1.0.pt")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

cfg = BertConfig.from_pretrained("bert-base-cased")
cfg._attn_implementation = "eager"
cfg.output_hidden_states = True
cfg.output_attentions = True

bert_model = BertForMaskedLM.from_pretrained("bert-base-cased", config=cfg)
bert_model.eval() 

z_q, z_k, z_o, z_mlp = None, None, None, None


ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
ds = ds.filter(lambda x: len(x["text"].split()) > 5 and not x["text"].startswith(" = "))

attn_module = bert_model.bert.encoder.layer[11].attention
intermediate_module = bert_model.bert.encoder.layer[11].intermediate
output_module = bert_model.bert.encoder.layer[11].output

Q_W, Q_b = attn_module.self.query.weight.detach(), attn_module.self.query.bias.detach()
K_W, K_b = attn_module.self.key.weight.detach(), attn_module.self.key.bias.detach()
V_W, V_b = attn_module.self.value.weight.detach(), attn_module.self.value.bias.detach()
O_W, O_b = attn_module.output.dense.weight.detach(), attn_module.output.dense.bias.detach()

W1, b1 = bert_model.bert.encoder.layer[11].intermediate.dense.weight.detach(), bert_model.bert.encoder.layer[11].intermediate.dense.bias.detach()
W2, b2 = bert_model.bert.encoder.layer[11].output.dense.weight.detach(), bert_model.bert.encoder.layer[11].output.dense.bias.detach()

# === ABLATION ===


def smooth(x: torch.tensor, n_features: int = 50, dim: int =-1):
    if x.ndim == 3:
        B, T, d = x.shape
    else:
        T, d = x.shape
    # find indices of top n_features by absolute value along last dim
    topk = torch.topk(x.abs(), k=n_features, dim=-1)
    # Create a mask of zeros, then scatter ones at the topk indices
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, topk.indices, True)
    x_ = x * mask

    return x_

def proj_hook(module, input, output, sae: SAE):
    X = input[0].detach()
    sparse_term, recon = sae(X)

    # sparse_term = smooth(sparse_term, n_features=n_features)

    global z_o
    z_o = sparse_term
    return sparse_term @ sae.S.T

def self_attn_hook(module, input, output):
    # input is a tuple of (hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
    X = input[0].detach()
    Q_s, Q_prime = q_t(X)
    K_s, K_prime = k_t(X)

    S_Q = q_t.S
    S_K = k_t.S

    B, T, d = X.shape
    dh = module.attention_head_size
    H = module.num_attention_heads

    # tQ, tK, tV = module.query(X), module.key(X), module.value(X)

    # === original version which does work ===

    # Q = Q_prime.view(B, T, H, dh).permute(0, 2, 1, 3)  # (B, H, T, dh)
    # K = K_prime.view(B, T, H, dh).permute(0, 2, 1, 3)  # (B, H, T, dh)

    # V = module.value(X).view(B, T, H, dh).permute(0, 2, 1, 3)  # (B, H, T, dh)

    # A = Q @ K.transpose(-1, -2) / (module.attention_head_size ** 0.5)
    # attn_probs = torch.nn.functional.softmax(A, dim=-1)
    # attn_probs = module.dropout(attn_probs)
    # context_layer = attn_probs @ V
    # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    # new_context_layer_shape = context_layer.size()[:-2] + (module.all_head_size,)
    # context_layer = context_layer.view(new_context_layer_shape)

    # return (context_layer, *output[1:])

    # === my version which does not work ===

    # Q_s, K_s shape (B, T, d)

    # Q_s = Q_s.masked_fill(Q_s.abs() <= 0.2, 0)
    # K_s = K_s.masked_fill(K_s.abs() <= 0.2, 0)

    # Q_s = smooth(Q_s, n_features=20)
    # K_s = smooth(K_s, n_features=20)

    global z_q, z_k
    z_q = Q_s
    z_k = K_s

    S_Q_T = S_Q.T.contiguous().reshape(d, H, dh).permute(1, 0, 2)  # (H, d, dh)
    S_K = S_K.contiguous().reshape(H, dh, d) # (H, dh, d)

    # === this does not work ===
    M = torch.einsum('hdf,hfe->hde', S_Q_T, S_K)  # (H, d, d)
    # [optional]
    # M = M.masked_fill(M.abs() <= 1, 0)

    # M = smooth(M, n_features=40, dim=-1)

    A_logits = torch.einsum('btd,hde,bse->bhts', Q_s, M, K_s)  # (H, T, T)

    # K_s_T = K_s.transpose(-1, -2)  # (B, d, T)

    # Q_s and K_s are shape (T, d). S_Q_T is shape (H, d, dh) and S_K is shape (H, dh, d). 
    # Q_p = torch.einsum('btd,hdf->bhtf', Q_s, S_Q_T)  # (B, H, T, dh)
    # K_p_T = torch.einsum('hfd,btd->bhft', S_K, K_s)  # (B, H, dh, T)
    # A_logits = Q_p @ K_p_T


    # Q_pr, K_pr = Q_prime.view(B, T, H, dh).permute(0, 2, 1, 3), K_prime.view(B, T, H, dh).permute(0, 2, 1, 3)
    # A_logits = Q_pr @ K_pr.transpose(-1, -2)

    # Q, K = module.query(X).view(B, T, H, dh).permute(0, 2, 1, 3), module.key(X).view(B, T, H, dh).permute(0, 2, 1, 3)
    # A_logits = Q @ K.transpose(-1, -2)

    # A_logits = Q_s.view(B, T, H, dh).permute(0, 2, 1, 3) @ K_s.view(B, T, H, dh).permute(0, 2, 3, 1)

    A_logits /= (dh ** 0.5)
    attn_probs = torch.nn.functional.softmax(A_logits, dim=-1)
    attn_probs = module.dropout(attn_probs)

    V = module.value(X).reshape(B, T, H, dh).permute(0, 2, 1, 3)  # (B, H, T, dh)

    context_layer = attn_probs @ V  # (B, H, T, dh)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.view(B, T, d)

    return (context_layer, *output[1:])

def mlp2_hook(module, input, output, sae: MLPSAE):
    sparse_term, recon = sae(input[0], W=W2, b=b2)

    # sparse_term = smooth(sparse_term, n_features=n_features)

    global z_mlp
    z_mlp = sparse_term
    return (sparse_term @ W2.T + b2) @ sae.S.T

# base_total_loss, base_n_masked = _run_dev_eval(bert_model, ds["validation"], batch_size=64, tokenizer=bert_tokenizer, device=torch.device("cpu"), max_length=128)
# print(f"Baseline: {base_total_loss / base_n_masked:.4f}")

# hq = attn_module.self.query.register_forward_hook(lambda module, input, output: proj_hook(module, input, output, q_t))
# hk = attn_module.self.key.register_forward_hook(lambda module, input, output: proj_hook(module, input, output, k_t))
h_self_attn = attn_module.self.register_forward_hook(self_attn_hook)

ho = attn_module.output.dense.register_forward_hook(lambda module, input, output: proj_hook(module, input, output, o_t))

h_mlp2 = bert_model.bert.encoder.layer[11].output.dense.register_forward_hook(lambda module, input, output: mlp2_hook(module, input, output, mlp_t))

# total_loss, n_masked = _run_dev_eval(bert_model, ds["validation"], batch_size=16, tokenizer=bert_tokenizer, device=torch.device("cpu"), max_length=128)
total_loss, n_masked = _run_dev_eval(bert_model, ds["validation"].select(list(range(1500))), batch_size=20, tokenizer=bert_tokenizer, device=torch.device("cpu"), max_length=128)

print(f"Baseline: 2.2813 | Ablated: {total_loss / n_masked:.4f}")

# hq.remove()
# hk.remove()
# hv.remove()
# ho.remove()

# h_self_attn.remove()

# h_mlp2.remove()

# quit()

# === THRESHOLDING M ===

# Compute the norms of the columns of QT_
# QT_ = q_t.T_.T
# col_norms = QT_.norm(dim=0)

# # Generate random samples whos norms follow the same distribution as the column norms of QT_
# mean_norm = col_norms.mean().item()
# std_norm = col_norms.std().item()
# alpha_samples = torch.normal(mean_norm, std_norm, size=(1000,))
# X = torch.randn(768, 1000)
# X = X * alpha_samples / X.norm(dim=0, keepdim=True)

# # Compute distribution of these dot products
# dot = X.T @ X
# dot = torch.tril(dot, diagonal=-1)
# mean, std = dot.mean().item(), dot.std().item()
# min_val, max_val = dot.min().item(), dot.max().item()

# print(f"Mean: {mean:.4f}, Std: {std:.4f}, Min: {min_val:.4f}, Max: {max_val:.4f}")

# quit()
# === NORMS ===
# norms_q = q_t.T_.norm(dim=0)
# norms_k = k_t.T_.norm(dim=0)
# norms_v = v_t.T_.norm(dim=0)
# norms_o = o_t.T_.norm(dim=0)

# plt.figure(figsize=(8, 6))
# plt.hist(norms_q.cpu().detach().numpy(), bins=50, alpha=0.6, label='Q', color='blue')
# plt.hist(norms_k.cpu().detach().numpy(), bins=50, alpha=0.6, label='K', color='orange')
# plt.hist(norms_v.cpu().detach().numpy(), bins=50, alpha=0.6, label='V', color='green')
# plt.hist(norms_o.cpu().detach().numpy(), bins=50, alpha=0.6, label='O', color='red')
# plt.legend()
# plt.xlabel("Norm Value")
# plt.ylabel("Frequency")
# plt.title("Histogram of Norms for Q, K, V, O")
# plt.show()

# === COS SIM ===
# V = q_t.T_
# norm = V.norm(dim=0, keepdim=True)
# cos_sim = (V.T @ V) / (norm.T @ norm)

# cos_sim_lower = torch.tril(cos_sim, diagonal=-1)
# CS = cos_sim_lower[cos_sim_lower != 0]
# print(f"Mean cosine similarity (lower triangle, off-diagonal): {CS.abs().max().item():.6f}")

# === Q-K atom similarities ===
# QK_atom_dot = (q_t.T_.T @ k_t.T_).abs()

# plt.figure(figsize=(8, 6))
# plt.imshow(QK_atom_dot[250:300, 250:300].cpu().detach().numpy(), cmap='Blues', aspect='auto')
# plt.colorbar(label='Dot Product Value')
# plt.title('Q-K Atom Dot Product Heatmap')
# plt.xlabel('K Atoms')
# plt.ylabel('Q Atoms')
# plt.show()


# === ATTENTION ===

sample = "The man loved his wife."

inputs = bert_tokenizer(sample, return_tensors="pt")
with torch.no_grad():
    outputs = bert_model(**inputs)
    X = outputs.hidden_states[10].squeeze(0)
    A = outputs.attentions[11].squeeze(0)

T, d = X.shape
dh = 64
H = 12

z_q, z_k = z_q.squeeze(0), z_k.squeeze(0)  # (T, d)
z_o, z_mlp = z_o.squeeze(0), z_mlp.squeeze(0)  # (T, d)


S_Q, S_K = q_t.S, k_t.S
S_Q_T = S_Q.T.contiguous().reshape(d, H, dh).permute(1, 0, 2)  # (H, d, dh)
S_K = S_K.contiguous().reshape(H, dh, d) # (H, dh, d)
M = torch.einsum('hdf,hfe->hde', S_Q_T, S_K)  # (H, d, d)
# [optional]
M = M.masked_fill(M.abs() <= 1, 0)

print()