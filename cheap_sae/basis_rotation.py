import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from datasets import load_dataset
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

cfg = BertConfig.from_pretrained("bert-base-cased")
cfg._attn_implementation = "eager"
model = BertModel.from_pretrained("bert-base-cased", config=cfg).to(device)
model.eval()  # we'll freeze BERT

for p in model.parameters():
    p.requires_grad_(False)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# clean ds a bit
ds = ds.filter(lambda x: len(x["text"].split()) > 5 and not x["text"].startswith(" = "))

batch_size = 32
learning_rate = 1e-3
num_epochs = 100
max_length = 128

# -------------------------
# Target: query projection at layer L
# -------------------------
layer: int = 6  # <- integer variable as requested (0-indexed layer)

bert_layer = model.encoder.layer[layer]
module = bert_layer.attention.self.query  # nn.Linear(hidden_size, all_head_size)
W = module.weight.detach()                # [out_dim, in_dim] = [hidden, hidden] for BERT
b = module.bias.detach()                  # [out_dim]

acts = {} # this will constantly be overwritten
def _hook(module, input, output):
    global acts
    # module_in is a tuple; for nn.Linear in BERT it's (X,) where X is [B, T, d]
    acts["x_in"] = input[0].detach()
    acts["y_out"] = output.detach()

handle = module.register_forward_hook(_hook)

d_out, d_in = W.shape
assert d_in == d_out, f"Expected square Q weight, got {W.shape}"
d = d_in

# -------------------------
# Learn U, V so that:
#   original:    Z = X W^T (+ b)
#   transformed: Z' = (V X) (U W)^T (+ b')  where b' = U b
#
# We enforce functional equivalence by training:
#   match:  (V X) (U W)^T + (U b)  ~=  X W^T + b
# and sparsity penalty on:
#   sparse: X (U W)^T
# -------------------------
class UVRotation(nn.Module):
    def __init__(self, d: int, init: str = "rand", eye_noise: float = 1e-3):
        super().__init__()
        if init == "eye":
            # Pure identity makes z_prime == z_orig exactly (and inv/ortho penalties minimal),
            # so most losses/gradients start at or near zero; add tiny noise to break symmetry.
            U0 = torch.eye(d) + eye_noise * torch.randn(d, d)
            V0 = torch.eye(d) + eye_noise * torch.randn(d, d)
        elif init == "rand":
            U0 = torch.randn(d, d) * 0.01 + torch.eye(d)
            V0 = torch.randn(d, d) * 0.01 + torch.eye(d)
        else:
            raise ValueError("init must be 'eye' or 'rand'")
        self.U = nn.Parameter(U0)
        self.V = nn.Parameter(V0)

    def forward(self, X: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        # X: [B, T, d]
        # W: [d, d] (out, in)
        UW = self.U @ W          # [d, d]
        Vx = X @ self.V.T        # [B, T, d]  (row-vector convention)
        z_prime = Vx @ UW.T      # [B, T, d]
        b_prime = b @ self.U.T   # [d]
        z_prime = z_prime + b_prime
        z_orig = X @ W.T + b
        sparse_term = X @ UW.T   # [B, T, d]
        return z_prime, z_orig, sparse_term, UW

uv = UVRotation(d=d, init="eye").to(device)

# Loss weights
lambda_sparse = 1   # increase to push more sparsity
lambda_inv = 0.0      # encourages U,V to be inverses
lambda_ortho = 0.0     # optional orthogonality regularizer

optimizer = torch.optim.AdamW(uv.parameters(), lr=learning_rate)

def token_batches(dataset_split, batch_size: int):
    texts = dataset_split["text"]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        yield {k: v.to(device) for k, v in enc.items()}

def orthogonality_penalty(A: torch.Tensor):
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    return ((A @ A.T - I) ** 2).mean()

# -------------------------
# Weights & Biases logging
# -------------------------

load_dotenv()  # loads from .env in CWD (and does nothing if missing)

# Ensure API key is available via env var
if not os.environ.get("WANDB_API_KEY"):
    raise RuntimeError(
        "WANDB_API_KEY not found in environment. "
        "Add it to your .env or export it before running."
    )

project = os.environ.get("WANDB_PROJECT", "cheap-sae")
entity = os.environ.get("WANDB_ENTITY", None)

run = wandb.init(
    project=project,
    entity=entity,
    name=f"bert-qproj-uv-rot-layer{layer}",
    job_type="train",
)

# Keep artifacts under ./cheap_sae/
root_dir = os.path.join(".", "cheap_sae")
artifacts_dir = os.path.join(root_dir, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

# Log config/hparams. (Update wandb config in-place without clobbering.)
wandb_config = {
    # data/model
    "model_name": "bert-base-cased",
    "dataset_name": "Salesforce/wikitext",
    "dataset_config": "wikitext-103-v1",
    "dataset_split": "train",
    "dataset_filter_min_words": 6,
    "dataset_filter_not_startswith": " = ",
    "layer": layer,
    "attn_implementation": getattr(cfg, "_attn_implementation", None),
    "device": str(device),
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "datasets_version": __import__("datasets").__version__,
    # training
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "max_length": max_length,
    "optimizer": "AdamW",
    # objective weights
    "lambda_sparse": lambda_sparse,
    "lambda_inv": lambda_inv,
    "lambda_ortho": lambda_ortho,
    # dimensions
    "d": d,
    "W_shape": tuple(W.shape),
    "b_shape": tuple(b.shape),
    # init
    "uv_init": "eye",
}
run.config.update(wandb_config, allow_val_change=True)

# -------------------------
# Training loop
# -------------------------
model_hidden_size = model.config.hidden_size
assert model_hidden_size == d

global_step = 0

for epoch in range(num_epochs):
    uv.train()
    running = {"loss": 0.0, "match": 0.0, "sparse": 0.0, "inv": 0.0, "ortho": 0.0}
    n_steps = 0

    for batch in tqdm(token_batches(ds["train"], batch_size), desc=f"epoch {epoch+1}/{num_epochs}"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Grab (input, output) at the exact module via a forward hook (no hidden_states).

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        X = acts["x_in"]     # [B, T, d] input to q_linear
        Y = acts["y_out"]    # [B, T, d] output of q_linear, i.e. X @ W.T + b

        # Run our transformation model on the captured pair for reconstruction
        z_prime, _, sparse_term, UW = uv(X, W, b)
        z_orig = Y

        # 1) Functional match
        match_loss = F.mse_loss(z_prime, z_orig)

        # 2) Sparsity on X (U W)^T
        sparse_loss = sparse_term.abs().mean()

        # 3) Encourage V ~ U^{-1}
        I = torch.eye(d, device=device)
        inv_loss = F.mse_loss(uv.V @ uv.U, I) + F.mse_loss(uv.U @ uv.V, I)

        # 4) Optional orthogonality-ish
        ortho_loss = (
            orthogonality_penalty(uv.U) + orthogonality_penalty(uv.V)
            if lambda_ortho > 0
            else torch.tensor(0.0, device=device)
        )

        loss = match_loss + lambda_sparse * sparse_loss + lambda_inv * inv_loss + lambda_ortho * ortho_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # per-step logging
        metrics = {
            "train/loss": float(loss.item()),
            "train/match_loss": float(match_loss.item()),
            "train/sparse_loss": float(sparse_loss.item()),
            "train/inv_loss": float(inv_loss.item()),
            "train/ortho_loss": float(ortho_loss.item()),
            "train/epoch": epoch + 1,
        }
        run.log(metrics, step=global_step)
        global_step += 1

        # epoch running averages
        running["loss"] += loss.item()
        running["match"] += match_loss.item()
        running["sparse"] += sparse_loss.item()
        running["inv"] += inv_loss.item()
        running["ortho"] += float(ortho_loss.item())
        n_steps += 1

    for k in running:
        running[k] /= max(n_steps, 1)

    print(
        f"epoch {epoch+1}: "
        f"loss={running['loss']:.6f} match={running['match']:.6f} "
        f"sparse={running['sparse']:.6f} inv={running['inv']:.6f}"
    )

    # per-epoch logging (averages)
    run.log(
        {
            "epoch/avg_loss": running["loss"],
            "epoch/avg_match_loss": running["match"],
            "epoch/avg_sparse_loss": running["sparse"],
            "epoch/avg_inv_loss": running["inv"],
            "epoch/avg_ortho_loss": running["ortho"],
            "epoch": epoch + 1,
        },
        step=global_step,
    )

# -------------------------
# Save + log learned transformed weights (U W)^T
# -------------------------
uv.eval()
with torch.no_grad():
    UW = (uv.U @ W).detach()          # [d, d]
    transformed_W_T = UW.T            # (U W)^T

    save_obj = {
        "layer": layer,
        "U": uv.U.detach().cpu(),
        "V": uv.V.detach().cpu(),
        "W": W.detach().cpu(),
        "b": b.detach().cpu(),
        "UW": UW.detach().cpu(),
        "transformed_W_T": transformed_W_T.detach().cpu(),
        "wandb_config": dict(run.config),
    }

    local_path = os.path.join(artifacts_dir, f"bert_qproj_layer{layer}_uv_rotation.pt")
    torch.save(save_obj, local_path)

# log to wandb as an artifact as well
artifact = wandb.Artifact(
    name=f"bert_qproj_layer{layer}_uv_rotation",
    type="uv_rotation",
    metadata={
        "layer": layer,
        "model_name": "bert-base-cased",
        "dataset_name": "Salesforce/wikitext",
        "dataset_config": "wikitext-103-v1",
        "d": d,
        "lambda_sparse": lambda_sparse,
        "lambda_inv": lambda_inv,
        "lambda_ortho": lambda_ortho,
    },
)
artifact.add_file(local_path)
run.log_artifact(artifact)

print(f"Saved locally to {local_path} and logged W&B artifact: {artifact.name}")