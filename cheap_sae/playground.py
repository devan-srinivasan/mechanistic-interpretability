import torch, json, numpy as np, nltk
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from helpers import Transformation, _run_dev_eval, SAE, MLPSAE
from datasets import load_dataset

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

def proj_hook(module, input, output, sae: SAE):
    X = input[0].detach()
    sparse_term, recon = sae(X)
    return recon

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

    # Q_s = Q_s.masked_fill(Q_s.abs() <= 0.1, 0)
    # K_s = K_s.masked_fill(K_s.abs() <= 0.1, 0)

    S_Q_T = S_Q.T.contiguous().reshape(d, H, dh).permute(1, 0, 2)  # (H, d, dh)
    S_K = S_K.contiguous().reshape(H, dh, d) # (H, dh, d)

    # === this does not work ===
    M = torch.einsum('hdf,hfe->hde', S_Q_T, S_K)  # (H, d, d)
    # [optional]
    # Ms = Ms.masked_fill(Ms.abs() <= 1.16, 0)

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

def mlp1_hook(module, input, output):
    return mlp_t.forward_1(output)

def mlp2_hook(module, input, output):
    return mlp_t.forward_2(input[0], W=W2, b=b2)

# base_total_loss, base_n_masked = _run_dev_eval(bert_model, ds["validation"], batch_size=64, tokenizer=bert_tokenizer, device=torch.device("cpu"), max_length=128)
# print(f"Baseline: {base_total_loss / base_n_masked:.4f}")

# hq = attn_module.self.query.register_forward_hook(lambda module, input, output: proj_hook(module, input, output, q_t))
# hk = attn_module.self.key.register_forward_hook(lambda module, input, output: proj_hook(module, input, output, k_t))
h_self_attn = attn_module.self.register_forward_hook(self_attn_hook)

ho = attn_module.output.dense.register_forward_hook(lambda module, input, output: proj_hook(module, input, output, o_t))

h_mlp1 = bert_model.bert.encoder.layer[11].intermediate.dense.register_forward_hook(mlp1_hook)
h_mlp2 = bert_model.bert.encoder.layer[11].output.dense.register_forward_hook(mlp2_hook)

# total_loss, n_masked = _run_dev_eval(bert_model, ds["validation"], batch_size=16, tokenizer=bert_tokenizer, device=torch.device("cpu"), max_length=128)
total_loss, n_masked = _run_dev_eval(bert_model, ds["validation"].select(list(range(128))), batch_size=10, tokenizer=bert_tokenizer, device=torch.device("cpu"), max_length=128)

print(f"True Baseline: 2.2813\nAblated: {total_loss / n_masked:.4f}")

# hq.remove()
# hk.remove()
# hv.remove()
ho.remove()

h_self_attn.remove()

h_mlp1.remove()
h_mlp2.remove()

quit()

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

Q_, Q, Q_sparse = q_t(X, W=Q_W, b=Q_b)
K_, K, K_sparse = q_t(X, W=K_W, b=K_b)
V_, V, V_sparse = v_t(X, W=V_W, b=V_b)

threshold = 0.1
Q_sparse = Q_sparse.masked_fill(Q_sparse.abs() <= threshold, 0)
K_sparse = K_sparse.masked_fill(K_sparse.abs() <= threshold, 0)

U_q = q_t.T_.T
U_v = v_t.T_.T
U_o = o_t.T_.T

# norms = U_q.norm(dim=1)
# print(norms.mean().item(), norms.std().item(), norms.min().item(), norms.max().item())

T, d = X.shape
dh = 64
H = 12

U_q_reshaped = U_q.T.view(d, H, dh)
M = torch.einsum('dhf,ehf->hde', U_q_reshaped, U_q_reshaped)  # (H, d, d)
M = M.masked_fill(M.abs() <= 1.16, 0)

h, i, j = 1, 4, 1
q_i, k_j = Q_sparse[i], K_sparse[j]
mixing = M[h]

k_j_mixed = mixing @ k_j
k_nt = torch.where(k_j.abs() > 0.1)[0]
k_mixed_nt = torch.where(k_j_mixed.abs() > 0.1)[0]

# separate features by those that have lots of mixing, those that are independent, and those that are suppressed
popular = torch.where((mixing > 0).sum(dim=-1) > 1)[0]
unpopular = torch.where((mixing > 0).sum(dim=-1) == 1)[0]
kernel = torch.where((mixing > 0).sum(dim=-1) == 0)[0]

attn_logits = torch.einsum('td,hde,se->hts', Q_sparse, M, K_sparse)  # (H, T, T)
attn_logits /= (dh ** 0.5)
attn_probs = torch.nn.functional.softmax(attn_logits, dim=-1)

U_v_reshaped = U_v.T.view(d, H, dh) # (d, H, dh)
context_layer = attn_probs @ torch.einsum('td,dhf->htf', V_sparse, U_v_reshaped) # (H, T, dh)
context_layer = context_layer.permute(1, 0, 2).contiguous().view(T, d) # (T, d)

O_, O, O_sparse = o_t(context_layer, W=O_W, b=O_b)

res_stream = O_sparse @ U_o.T
# print(res_stream.shape)

# print(f"Q_sparse: {Q_sparse.abs().mean().item():.4f} {Q_sparse.abs().std().item():.4f} {Q_sparse.abs().min().item():.4f} {Q_sparse.abs().max().item():.4f}")
# print(f"K_sparse: {K_sparse.abs().mean().item():.4f} {K_sparse.abs().std().item():.4f} {K_sparse.abs().min().item():.4f} {K_sparse.abs().max().item():.4f}")
# print(f"V_sparse: {V_sparse.abs().mean().item():.4f} {V_sparse.abs().std().item():.4f} {V_sparse.abs().min().item():.4f} {V_sparse.abs().max().item():.4f}")
# print(f"O_sparse: {O_sparse.abs().mean().item():.4f} {O_sparse.abs().std().item():.4f} {O_sparse.abs().min().item():.4f} {O_sparse.abs().max().item():.4f}")

# Use a diverging colormap with a symmetric log normalization
# import matplotlib.colors as mcolors
# for h in range(H):
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     norm = mcolors.SymLogNorm(linthresh=1e-2, linscale=1.0, vmin=None, vmax=None, base=10)
#     cmap = "seismic"

#     im0 = axes[0].imshow((attn_logits[h] * (dh ** 0.5)).cpu().detach().numpy(), cmap=cmap, norm=norm, aspect='auto')
#     axes[0].set_title(f"Head {h}")
#     axes[0].set_xlabel("Key Position")
#     axes[0].set_ylabel("Query Position")
#     fig.colorbar(im0, ax=axes[0])

#     im1 = axes[1].imshow((Q_sparse @ K_sparse.transpose(-1,-2)).cpu().detach().numpy(), cmap=cmap, norm=norm, aspect='auto')
#     axes[1].set_title(f"Head {h} (new)")
#     axes[1].set_xlabel("Key Position")
#     axes[1].set_ylabel("Query Position")
#     fig.colorbar(im1, ax=axes[1])

#     plt.tight_layout()
#     plt.show()
