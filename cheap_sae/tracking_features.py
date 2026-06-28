import torch, json, numpy as np, nltk, os
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertForQuestionAnswering
from helpers import Transformation, _run_dev_eval, SAE, MLPSAE, token_batches, token_batches_squad
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

run_path = "cheap_sae/features/squad"

device = "cuda:7" if torch.cuda.is_available() else "cpu"

batch_idx = None

def load_model(sae, path):
    save = torch.load(path)
    sae.U.data.copy_(save["U"])
    sae.S.data.copy_(save["S"])
    return sae

q_sae = load_model(SAE(d=768, init="eye"), "cheap_sae/artifacts/sae_layer11_correct/sae_q_layer11_sparse1.0_inv1.0_rel_match1.0.pt")
k_sae = load_model(SAE(d=768, init="eye"), "cheap_sae/artifacts/sae_layer11_correct/sae_k_layer11_sparse1.0_inv1.0_rel_match1.0.pt")
o_sae = load_model(SAE(d=768, init="eye"), "cheap_sae/artifacts/sae_layer11_correct/sae_o_layer11_sparse1.0_inv1.0_rel_match1.0.pt")
mlp_sae = load_model(MLPSAE(d1=3072, d2=768, init="eye"), "cheap_sae/artifacts/sae_layer11_correct/mlp_sae_mlp_layer11_sparse1.0_rel_match1.0.pt")

# ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
# ds = ds.filter(lambda x: len(x["text"].split()) > 5 and not x["text"].startswith(" = "))

ds = load_from_disk("cheap_sae/data/bert_cased_squad_tokenized")["validation"]

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

cfg = BertConfig.from_pretrained("bert-base-cased")
cfg._attn_implementation = "eager"
cfg.output_hidden_states = True
cfg.output_attentions = True

model = BertForQuestionAnswering.from_pretrained("cheap_sae/models/squad", config=cfg)
model.eval()

attn_module = model.bert.encoder.layer[11].attention
intermediate_module = model.bert.encoder.layer[11].intermediate
output_module = model.bert.encoder.layer[11].output

Q_W, Q_b = attn_module.self.query.weight.detach(), attn_module.self.query.bias.detach()
K_W, K_b = attn_module.self.key.weight.detach(), attn_module.self.key.bias.detach()
O_W, O_b = attn_module.output.dense.weight.detach(), attn_module.output.dense.bias.detach()

W1, b1 = model.bert.encoder.layer[11].intermediate.dense.weight.detach(), model.bert.encoder.layer[11].intermediate.dense.bias.detach()
W2, b2 = model.bert.encoder.layer[11].output.dense.weight.detach(), model.bert.encoder.layer[11].output.dense.bias.detach()


def proj_hook(module, input, output, sae: SAE):
    global batch_idx
    X = input[0].detach()
    sparse_term, recon = sae(X)

    # TODO save sparse_term somehow
    torch.save(sparse_term, os.path.join(run_path, str(batch_idx), "o.pt"))

    return recon

def self_attn_hook(module, input, output):
    global batch_idx
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
    torch.save(Q_s.cpu(), os.path.join(run_path, str(batch_idx), "Q_s.pt"))
    torch.save(K_s.cpu(), os.path.join(run_path, str(batch_idx), "K_s.pt"))

    S_Q_T = S_Q.T.contiguous().reshape(d, H, dh).permute(1, 0, 2)  # (H, d, dh)
    S_K = S_K.contiguous().reshape(H, dh, d) # (H, dh, d)

    # === this does not work ===
    M = torch.einsum('hdf,hfe->hde', S_Q_T, S_K)  # (H, d, d)
    # [optional]
    # M = M.masked_fill(M.abs() <= 1, 0)
    
    # TODO save M if not saved already
    if not os.path.exists(run_path + "M.pt"):
        torch.save(M, run_path + "M.pt")

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
    global batch_idx
    sparse_term, recon = mlp_sae(input[0], W=W2, b=b2)

    # TODO save sparse_term somehow
    torch.save(sparse_term, os.path.join(run_path, str(batch_idx), "mlp2.pt"))

    return recon

h_self_attn = attn_module.self.register_forward_hook(self_attn_hook)
ho = attn_module.output.dense.register_forward_hook(lambda module, input, output: proj_hook(module, input, output, o_sae))
h_mlp2 = model.bert.encoder.layer[11].output.dense.register_forward_hook(mlp2_hook)

def measure_dir_size(dir: str):
    """ Measures the size in MB of a directory (Recursively) """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)
    
# pbar = tqdm(enumerate(token_batches(ds, 32, tokenizer, device, 128)))
bs = 32
pbar = tqdm(enumerate(token_batches_squad(ds, bs, tokenizer, device, 128)), total=len(ds) // bs)
for batch_idx, batch in pbar:
    os.makedirs(os.path.join(run_path, str(batch_idx)), exist_ok=True)
    tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch['input_ids']]
    start_positions = batch['start_positions'].cpu().tolist()
    end_positions = batch['end_positions'].cpu().tolist()
    with open(os.path.join(run_path, str(batch_idx), "tokens.json"), "w") as f:
        json.dump({
            "tokens": tokens,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }, f)
    with torch.no_grad():
        # _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], start_positions=batch['start_positions'], end_positions=batch['end_positions'])
    
    pbar.set_postfix_str(f"Size: {measure_dir_size(run_path):.2f} MB")
    quit()
