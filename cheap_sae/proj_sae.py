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
from helpers import token_batches, _run_dev_eval, SAE
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
    p.add_argument("--name", type=str, default=None, help="Name for this run (defaults to basis_rotation_layer{L}_lambda_sparse{lambda_sparse}_lambda_inv{lambda_inv})_lambda_rel_match{lambda_rel_match})")

    p.add_argument("--layer", type=int, default=6, help="0-indexed BERT layer")

    p.add_argument("--lambda_sparse", type=float, default=1.0)
    p.add_argument("--lambda_inv", type=float, default=1.0)
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
module = bert_layer.attention.output.dense  # nn.Linear(hidden_size, all_head_size)
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

sae = SAE(d=d, init="rand").to(args.device)

optimizer = torch.optim.AdamW(sae.parameters(), lr=args.learning_rate)

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
    args.name = f"bert_oproj_layer{args.layer}_transformation_lambda_sparse{args.lambda_sparse}_lambda_inv{args.lambda_inv}_lambda_rel_match{args.lambda_rel_match}"

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
    "lambda_inv": args.lambda_inv,
    "lambda_rel_match": args.lambda_rel_match,
    # dimensions
    "d": d,
    # init
    "transformation_init": "rand",
}
run.config.update(wandb_config, allow_val_change=True)

# -------------------------
# Training loop
# -------------------------
model_hidden_size = model.config.hidden_size
assert model_hidden_size == d

global_step = 0

dev_base_MLM = None

for epoch in range(args.num_epochs):
    sae.train()
    running = {"loss": 0.0, "match": 0.0, "sparse": 0.0, "inv": 0.0, "rel_match": 0.0}
    n_steps = 0

    for batch in tqdm(token_batches(ds["train"], args.batch_size, tokenizer, args.device, args.max_length), desc=f"epoch {epoch+1}/{args.num_epochs} n_batches={len(ds['train'])//args.batch_size}"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Grab (input, output) at the exact module via a forward hook (no hidden_states).

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        X = acts["x_in"]     # [B, T, d] input to q_linear
        Y = acts["y_out"]    # [B, T, d] output of q_linear, i.e. X @ W.T + b
        acts = {}  # clear for next step

        # Run our transformation model on the captured pair for reconstruction
        sparse_term, z_prime = sae(X)
        z_orig = Y

        # 1) Functional match
        match_loss = F.mse_loss(z_prime, z_orig)
        rel_match_loss = torch.norm(z_prime - z_orig) / torch.norm(z_orig)  # MSE + relative error, to encourage both absolute and relative closeness

        # 2) Sparsity on X (U W)^T
        sparse_loss = sparse_term.abs().mean()

        # 3) Optional invertibility loss
        inv_loss = F.mse_loss(sae.U @ sae.S.T, torch.eye(d, device=sae.U.device))

        loss = match_loss + args.lambda_rel_match * rel_match_loss + args.lambda_sparse * sparse_loss + args.lambda_inv * inv_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # per-step logging
        metrics = {
            "train/loss": float(loss.item()),
            "train/match_loss": float(match_loss.item()),
            "train/rel_match_loss": float(rel_match_loss.item()),
            "train/sparse_loss": float(sparse_loss.item()),
            "train/inv_loss": float(inv_loss.item()),
            "train/epoch": epoch + 1,
        }
        run.log(metrics, step=global_step)
        global_step += 1

        # epoch running averages
        running["loss"] += loss.item()
        running["match"] += match_loss.item()
        running["rel_match"] += rel_match_loss.item()
        running["sparse"] += sparse_loss.item()
        running["inv"] += float(inv_loss.item())
        n_steps += 1

    for k in running:
        running[k] /= max(n_steps, 1)

    print(
        f"epoch {epoch+1}: "
        f"loss={running['loss']:.6f} match={running['match']:.6f} rel_match={running['rel_match']:.6f} "
        f"sparse={running['sparse']:.6f} inv={running['inv']:.6f}"
    )

    # per-epoch logging (averages)

    # EVAL

    def _eval_hook(module, input, output):
        X = input[0].detach()

        _, z_prime = sae(X)

        return z_prime.to(output.device)
    
    # -------------------------
    # EVAL on dev set (validation)
    # -------------------------
    try:
        sae.eval()
        mlm_model = BertForMaskedLM.from_pretrained("bert-base-cased").to(args.device)
        mlm_model.eval()

        eval_handle = mlm_model.bert.encoder.layer[args.layer].attention.output.dense.register_forward_hook(_eval_hook)
        dev_mlm = _run_dev_eval(mlm_model, ds["validation"], args.batch_size, tokenizer, args.device, args.max_length)
        eval_handle.remove()

        # If baseline not computed yet, run dev eval again WITHOUT the eval hook, store baseline, and log
        if dev_base_MLM is None:
            dev_base_MLM = _run_dev_eval(mlm_model, ds["validation"], args.batch_size, tokenizer, args.device, args.max_length)
    except Exception as e:
        print(f"Error during dev evaluation: {e}")
        dev_mlm = None

    run.log(
        {
            "epoch/avg_loss": running["loss"],
            "epoch/avg_match_loss": running["match"],
            "epoch/avg_rel_match_loss": running["rel_match"],
            "epoch/avg_sparse_loss": running["sparse"],
            "epoch/avg_inv_loss": running["inv"],
            "epoch/dev_mlm": dev_mlm,
            "epoch/dev_base_mlm": dev_base_MLM,
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
            "U": sae.U.detach().cpu(),
            "S": sae.S.detach().cpu(),
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
            "d": d,
            "lambda_sparse": args.lambda_sparse,
            "lambda_inv": args.lambda_inv,
            "lambda_rel_match": args.lambda_rel_match,
        },
    )
    artifact.add_file(local_path)
    run.log_artifact(artifact)

    print(f"Saved locally to {local_path} and logged W&B artifact: {artifact.name}")