import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig, BertForMaskedLM
from datasets import load_dataset
from tqdm import tqdm
import wandb
from dotenv import load_dotenv
from helpers import token_batches, _run_dev_eval, MLPSAE
import argparse

# -------------------------
# Setup
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="")
    p.add_argument("--device", type=str, default=None, help='e.g. "cuda:6", "mps", or "cpu" (default: auto)')
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--name", type=str, default=None, help="Name for this run (defaults to basis_rotation_layer{L}_sparse{lambda_sparse}_rel_match{lambda_rel_match})")

    p.add_argument("--layer", type=int, default=6, help="0-indexed BERT layer")
    p.add_argument("--module", type=str, default="q", help="Which attention module to target: q, k, v, o, mlp1, mlp2")

    p.add_argument("--lambda_sparse", type=float, default=1.0)
    p.add_argument("--lambda_rel_match", type=float, default=1.0)
    return p.parse_args()

args = parse_args()

torch.set_float32_matmul_precision("high")

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

cfg = BertConfig.from_pretrained("bert-base-cased")
cfg._attn_implementation = "eager"
model = BertModel.from_pretrained("bert-base-cased", config=cfg).to(args.device)
model.eval()  # we'll freeze BERT

for p in model.parameters():
    p.requires_grad_(False)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# clean ds a bit
ds = ds.filter(lambda x: len(x["text"].split()) > 5 and not x["text"].startswith(" = "))

# For local testing
if torch.mps.is_available():
    ds['train'] = ds['train'].shuffle(seed=42).select(range(64))
    ds['validation'] = ds['validation'].shuffle(seed=42).select(range(16))
    args.batch_size = 4

# -------------------------
# Target: some projection at layer L
# -------------------------

bert_layer = model.encoder.layer[args.layer]
if args.module == "mlp":
    module1 = bert_layer.intermediate.dense
    module2 = bert_layer.output.dense
else:
    raise ValueError(f"Invalid module {args.module}, must be mlp")

acts = {} # this will constantly be overwritten
def _hook1(module, input, output):
    global acts
    # module_in is a tuple; for nn.Linear in BERT it's (X,) where X is [B, T, d]
    acts["x1_in"] = input[0].detach()
    acts["y1_out"] = output.detach()

def _hook2(module, input, output):
    global acts
    # module_in is a tuple; for nn.Linear in BERT it's (X,) where X is [B, T, d]
    acts["x2_in"] = input[0].detach()
    acts["y2_out"] = output.detach()

handle1 = module1.register_forward_hook(_hook1)
handle2 = module2.register_forward_hook(_hook2)

W1, b1 = module1.weight.detach(), module1.bias.detach()
W2, b2 = module2.weight.detach(), module2.bias.detach()

d1 = W1.shape[0]
d2 = W2.shape[0]

mlp_sae = MLPSAE(d1=d1, d2=d2, init="rand").to(args.device)

gelu = bert_layer.intermediate.intermediate_act_fn  # get the GELU from the original model to apply to the sparse term
# Freeze GELU if it has parameters (it usually doesn't, but for safety)
for p in gelu.parameters() if hasattr(gelu, "parameters") else []:
    p.requires_grad_(False)

optimizer = torch.optim.AdamW(mlp_sae.parameters(), lr=args.learning_rate)

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

if not args.name:
    args.name = f"mlp_sae_{args.module}_layer{args.layer}_sparse{args.lambda_sparse}_rel_match{args.lambda_rel_match}"

run = wandb.init(
    project=project,
    entity=entity,
    name=args.name,
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
    "layer": args.layer,
    "attn_implementation": getattr(cfg, "_attn_implementation", None),
    "device": str(args.device),
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "datasets_version": __import__("datasets").__version__,
    # training
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "num_epochs": args.num_epochs,
    "max_length": args.max_length,
    "optimizer": "AdamW",
    # objective weights
    "lambda_sparse": args.lambda_sparse,
    "lambda_rel_match": args.lambda_rel_match,
    # dimensions
    "d1": d1,
    "d2": d2,
    # init
    "transformation_init": "rand",
}
run.config.update(wandb_config, allow_val_change=True)

# -------------------------
# Training loop
# -------------------------

global_step = 0

dev_base_MLM = None

for epoch in range(args.num_epochs):
    mlp_sae.train()
    running = {"loss": 0.0, "match": 0.0, "sparse": 0.0, "rel_match": 0.0}
    n_steps = 0

    for batch in tqdm(token_batches(ds["train"], args.batch_size, tokenizer, args.device, args.max_length), desc=f"epoch {epoch+1}/{args.num_epochs} n_batches={len(ds['train'])//args.batch_size}"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Grab (input, output) at the exact module via a forward hook (no hidden_states).

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        X = acts["x2_in"]  # [B, T, d1]
        z_orig = acts["y2_out"]  # [B, T, d2]
        sparse_term, z_prime = mlp_sae(X, W=W2, b=b2)  # [B, T, d1], [B, T, d2]
        
        acts = {}  # clear for next step

        # Run our transformation model on the captured pair for reconstruction

        # 1) Functional match
        match_loss = F.mse_loss(z_prime, z_orig)
        rel_match_loss = torch.norm(z_prime - z_orig) / torch.norm(z_orig)  # MSE + relative error, to encourage both absolute and relative closeness

        # 2) Sparsity on X (U W)^T
        sparse_loss = sparse_term.abs().mean()

        loss = match_loss + args.lambda_rel_match * rel_match_loss + args.lambda_sparse * sparse_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # per-step logging
        metrics = {
            "train/loss": float(loss.item()),
            "train/match_loss": float(match_loss.item()),
            "train/rel_match_loss": float(rel_match_loss.item()),
            "train/sparse_loss": float(sparse_loss.item()),
            "train/epoch": epoch + 1,
        }
        run.log(metrics, step=global_step)
        global_step += 1

        # epoch running averages
        running["loss"] += loss.item()
        running["match"] += match_loss.item()
        running["rel_match"] += rel_match_loss.item()
        running["sparse"] += sparse_loss.item()
        n_steps += 1

    for k in running:
        running[k] /= max(n_steps, 1)

    print(
        f"epoch {epoch+1}: "
        f"loss={running['loss']:.6f} match={running['match']:.6f} rel_match={running['rel_match']:.6f} "
        f"sparse={running['sparse']:.6f}"
    )

    # per-epoch logging (averages)

    # EVAL

    run.log(
        {
            "epoch/avg_loss": running["loss"],
            "epoch/avg_match_loss": running["match"],
            "epoch/avg_rel_match_loss": running["rel_match"],
            "epoch/avg_sparse_loss": running["sparse"],
            "epoch": epoch + 1,
        },
        step=global_step,
    )

    # -------------------------
    # Save + log learned transformed weights (U W)^T
    # -------------------------
    
    with torch.no_grad():
        save_obj = {
            "layer": args.layer,
            "U": mlp_sae.U.detach().cpu(),
            "S": mlp_sae.S.detach().cpu(),
            "wandb_config": dict(run.config),
        }

        local_path = os.path.join(artifacts_dir, f"{args.name}.pt")
        torch.save(save_obj, local_path)

    # log to wandb as an artifact as well
    artifact = wandb.Artifact(
        name=args.name,
        type="transformation",
        metadata={
            "layer": args.layer,
            "model_name": "bert-base-cased",
            "dataset_name": "Salesforce/wikitext",
            "dataset_config": "wikitext-103-v1",
            "d1": d1,
            "d2": d2,
            "lambda_sparse": args.lambda_sparse,
            "lambda_rel_match": args.lambda_rel_match,
        },
    )
    artifact.add_file(local_path)
    run.log_artifact(artifact)

    print(f"Saved locally to {local_path} and logged W&B artifact: {artifact.name}")