import torch, json, numpy as np, nltk
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from helpers import Transformation, _run_dev_eval
from datasets import load_dataset

def load_model(path):
    model_save = torch.load(path, map_location="cpu")
    T, T_ = model_save['T'], model_save['T_']
    transformation = Transformation(d=T.shape[0], init="rand")
    transformation.T.data.copy_(T)
    transformation.T_.data.copy_(T_)
    transformation.eval()
    return transformation

q_t = load_model("cheap_sae/artifacts/layer11/bert_qproj_layer11_transformation_lambda_sparse1.0_lambda_inv0.0_lambda_rel_match0.0.pt")
k_t = load_model("cheap_sae/artifacts/layer11/bert_kproj_layer11_transformation_lambda_sparse1.0_lambda_inv0.0_lambda_rel_match0.0.pt")
v_t = load_model("cheap_sae/artifacts/layer11/bert_vproj_layer11_transformation_lambda_sparse1.0_lambda_inv0.0_lambda_rel_match0.0.pt")
o_t = load_model("cheap_sae/artifacts/layer11/bert_oproj_layer11_transformation_lambda_sparse1.0_lambda_inv0.0_lambda_rel_match0.0.pt")


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

cfg = BertConfig.from_pretrained("bert-base-cased")
cfg._attn_implementation = "eager"

bert_model = BertForMaskedLM.from_pretrained("bert-base-cased", config=cfg)
bert_model.eval() 


ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
ds = ds.filter(lambda x: len(x["text"].split()) > 5 and not x["text"].startswith(" = "))

module = bert_model.bert.encoder.layer[11].attention

Q_W, Q_b = module.self.query.weight.detach(), module.self.query.bias.detach()
K_W, K_b = module.self.key.weight.detach(), module.self.key.bias.detach()
V_W, V_b = module.self.value.weight.detach(), module.self.value.bias.detach()
O_W, O_b = module.output.dense.weight.detach(), module.output.dense.bias.detach()


# === ABLATION ===

def q_hook(module, input, output):
    X = input[0].detach()
    z_prime, z_orig, sparse_term = q_t(X, W=Q_W, b=torch.zeros_like(Q_b))
    return sparse_term @ q_t.T_.to(output.device)

def k_hook(module, input, output):
    X = input[0].detach()
    # z_prime, z_orig, sparse_term = k_t(X, W=K_W, b=torch.zeros_like(K_b))
    z_prime, z_orig, sparse_term = q_t(X, W=K_W, b=K_b)
    return sparse_term @ q_t.T_.to(output.device)

def v_hook(module, input, output):
    X = input[0].detach()
    z_prime, z_orig, sparse_term = v_t(X, W=V_W, b=V_b)
    # z_prime, z_orig, sparse_term = q_t(X, W=V_W, b=torch.zeros_like(V_b))
    return sparse_term @ v_t.T_.to(output.device)

def o_hook(module, input, output):
    X = input[0].detach()
    z_prime, z_orig, sparse_term = o_t(X, W=O_W, b=O_b)
    # z_prime, z_orig, sparse_term = q_t(X, W=O_W, b=torch.zeros_like(O_b))
    return sparse_term @ o_t.T_.to(output.device)

def self_attn_hook(module, input, output):
    # input is a tuple of (hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
    X = input[0].detach()
    Q_prime, Q_orig, Q_sparse = q_t(X, W=Q_W, b=Q_b)
    K_prime, K_orig, K_sparse = q_t(X, W=K_W, b=K_b)
    V_prime, V_orig, V_sparse = v_t(X, W=V_W, b=V_b)

    def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (module.num_attention_heads, module.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    QT_ = q_t.T_.to(X.device)
    VT_ = v_t.T_.to(X.device)

    # tQ, tK, tV = module.query(X), module.key(X), module.value(X)
    
    # Q = Q_sparse
    # K = K_sparse
    # V = V_sparse @ VT_

    # Q = transpose_for_scores(Q_prime)
    # K = transpose_for_scores(K_prime)
    V = transpose_for_scores(V_prime)

    # A = Q @ K.transpose(-1, -2) / (module.attention_head_size ** 0.5)
    # attn_probs = torch.nn.functional.softmax(A, dim=-1)
    # attn_probs = module.dropout(attn_probs)
    # context_layer = attn_probs @ V
    # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    # new_context_layer_shape = context_layer.size()[:-2] + (module.all_head_size,)
    # context_layer = context_layer.view(new_context_layer_shape)

    # === my version ===
    B, T, d = X.shape
    dh = module.attention_head_size
    H = module.num_attention_heads

    Q_s, K_s = Q_sparse, K_sparse # shape (B, T, d)
    QT_ = QT_.view(d, H, dh) # (d, H, dh)

    # === this works
    # Qh = torch.einsum('btd,dhf->bhtf', Q_s, QT_)
    # Kh = torch.einsum('btd,dhf->bhtf', K_s, QT_)
    # attn_probs = torch.einsum('bhtf,bhsf->bhts', Qh, Kh) # (B, H, T, T)
    # ===

    # === this now works too ===
    Ms = torch.einsum('dhf,ehf->hde', QT_, QT_)          # (H, d, d)
    attn_probs = torch.einsum('btd,hde,bse->bhts', Q_s, Ms, K_s)  # (B, H, T, T)

    attn_probs /= (dh ** 0.5)
    attn_probs = torch.nn.functional.softmax(attn_probs, dim=-1)
    attn_probs = module.dropout(attn_probs)

    context_layer = attn_probs @ V  # (B, H, T, dh)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.view(B, T, d)

    return (context_layer, *output[1:])

# base_total_loss, base_n_masked = _run_dev_eval(bert_model, ds["validation"], batch_size=64, tokenizer=bert_tokenizer, device=torch.device("cpu"), max_length=128)
# print(f"Baseline: {base_total_loss / base_n_masked:.4f}")

# hq = module.self.query.register_forward_hook(q_hook)
# hk = module.self.key.register_forward_hook(k_hook)
# hv = module.self.value.register_forward_hook(v_hook)
h_self_attn = module.self.register_forward_hook(self_attn_hook)

ho = module.output.dense.register_forward_hook(o_hook)

# total_loss, n_masked = _run_dev_eval(bert_model, ds["validation"], batch_size=64, tokenizer=bert_tokenizer, device=torch.device("cpu"), max_length=128)
total_loss, n_masked = _run_dev_eval(bert_model, ds["validation"].select(list(range(64))), batch_size=10, tokenizer=bert_tokenizer, device=torch.device("cpu"), max_length=128)

print(f"Baseline: 2.2813\nAblated: {total_loss / n_masked:.4f}")

# hq.remove()
# hk.remove()
# hv.remove()
ho.remove()

h_self_attn.remove()

quit()


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

Q_, Q, Q_sparse = q_t(X, W=Q_W, b=Q_b)
K_, K, K_sparse = q_t(X, W=K_W, b=K_b)
V_, V, V_sparse = v_t(X, W=V_W, b=V_b)
O_, O, O_sparse = o_t(X, W=O_W, b=O_b)

A = Q @ K.T

Q_sparse = Q_sparse.masked_fill(Q_sparse.abs() <= 0.1, 0)
K_sparse = K_sparse.masked_fill(K_sparse.abs() <= 0.1, 0)

T_ = q_t.T_

alpha_factor = T_ @ T_.T
alpha_factor = alpha_factor.masked_fill(alpha_factor.abs() <= 4, 0)

A_ = Q_sparse @ alpha_factor @ K_sparse.T
# A_ = Q_ @ K_.T

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

im0 = axs[0].imshow(A.cpu().detach().numpy(), cmap='viridis', aspect='auto')
axs[0].set_title("A")
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(A_.cpu().detach().numpy(), cmap='viridis', aspect='auto')
axs[1].set_title("A_ (approximated)")
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

