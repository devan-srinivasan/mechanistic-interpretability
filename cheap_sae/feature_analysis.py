import torch, json, numpy as np, nltk
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering
from helpers import SAE, MLPSAE

def load_model(sae, path):
    save = torch.load(path)
    sae.U.data.copy_(save["U"])
    sae.S.data.copy_(save["S"])
    return sae

q_sae = load_model(SAE(d=768, init="eye"), "cheap_sae/artifacts/sae_layer11_correct/sae_q_layer11_sparse1.0_inv1.0_rel_match1.0.pt")
k_sae = load_model(SAE(d=768, init="eye"), "cheap_sae/artifacts/sae_layer11_correct/sae_k_layer11_sparse1.0_inv1.0_rel_match1.0.pt")
o_sae = load_model(SAE(d=768, init="eye"), "cheap_sae/artifacts/sae_layer11_correct/sae_o_layer11_sparse1.0_inv1.0_rel_match1.0.pt")
mlp_sae = load_model(MLPSAE(d1=3072, d2=768, init="eye"), "cheap_sae/artifacts/sae_layer11_correct/mlp_sae_mlp_layer11_sparse1.0_rel_match1.0.pt")

model = BertForQuestionAnswering.from_pretrained("cheap_sae/models/squad")

(start_dir, end_dir), b = model.qa_outputs.weight.detach(), model.qa_outputs.bias.detach()  # W shape (2, 768), b shape (2,)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

start_logit_feature = model.qa_outputs.weight[0]  # shape (768,)
end_logit_feature = model.qa_outputs.weight[1]  # shape (768,)

W2, b2 = model.bert.encoder.layer[11].output.dense.weight.detach(), model.bert.encoder.layer[11].output.dense.bias.detach()

z_q, z_k, z_mlp, z_o, mix = None, None, None, None, None

Q_atoms, K_atoms = q_sae.S.detach().T, k_sae.S.detach().T
O_atoms = o_sae.S.detach().T
MLP_atoms = W2.T @ mlp_sae.S.detach().T


def proj_hook(module, input, output):
    global z_o
    X = input[0].detach()
    sparse_term, recon = o_sae(X)
    z_o = sparse_term
    return recon

def self_attn_hook(module, input, output):
    global z_q, z_k, mix
    # input is a tuple of (hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
    X = input[0].detach()
    Q_s, Q_prime = q_sae(X)
    K_s, K_prime = k_sae(X)

    S_Q = q_sae.S
    S_K = k_sae.S

    B, T, d = X.shape
    dh = module.attention_head_size
    H = module.num_attention_heads

    # Q_s = Q_s.masked_fill(Q_s.abs() <= 0.2, 0)
    # K_s = K_s.masked_fill(K_s.abs() <= 0.2, 0)

    # TODO save Q_s and K_s somehow
    z_q, z_k = Q_s, K_s

    S_Q_T = S_Q.T.contiguous().reshape(d, H, dh).permute(1, 0, 2)  # (H, d, dh)
    S_K = S_K.contiguous().reshape(H, dh, d) # (H, dh, d)

    # === this does not work ===
    M = torch.einsum('hdf,hfe->hde', S_Q_T, S_K)  # (H, d, d)
    # [optional]
    # M = M.masked_fill(M.abs() <= 1, 0)
    mix = M

    A_logits = torch.einsum('btd,hde,bse->bhts', Q_s, M, K_s)  # (H, T, T)

    A_logits /= (dh ** 0.5)
    attn_probs = torch.nn.functional.softmax(A_logits, dim=-1)
    attn_probs = module.dropout(attn_probs)

    V = module.value(X).reshape(B, T, H, dh).permute(0, 2, 1, 3)  # (B, H, T, dh)

    context_layer = attn_probs @ V  # (B, H, T, dh)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.view(B, T, d)

    return (context_layer, *output[1:])

def mlp2_hook(module, input, output):
    global z_mlp
    sparse_term, recon = mlp_sae(input[0], W=W2, b=b2)
    z_mlp = sparse_term
    return recon

attn_module = model.bert.encoder.layer[11].attention
intermediate_module = model.bert.encoder.layer[11].intermediate
output_module = model.bert.encoder.layer[11].output

h_self_attn = attn_module.self.register_forward_hook(self_attn_hook)
ho = attn_module.output.dense.register_forward_hook(proj_hook)
h_mlp2 = output_module.dense.register_forward_hook(mlp2_hook)

context = "Soccer is the best sport in the world. Basketball is not the best sport in the world."
question = "What is the best sport in the world?"

enc = tokenizer(context, question, return_tensors="pt")

with torch.no_grad():
    outputs = model(**enc)
    start_logits = outputs.start_logits     # shape (1, seq_len)
    end_logits = outputs.end_logits         # shape (1, seq_len)

z_q = z_q.squeeze(0)
z_k = z_k.squeeze(0)
z_o = z_o.squeeze(0)
z_mlp = z_mlp.squeeze(0)

s, e = torch.argmax(start_logits), torch.argmax(end_logits)

print("pred:", tokenizer.decode(enc["input_ids"][0][s:e+1]))