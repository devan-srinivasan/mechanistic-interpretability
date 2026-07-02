from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers.pipelines.question_answering import select_starts_ends
import torch, torch.nn as nn, numpy as np

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

class Transformation(nn.Module):
    def __init__(self, d: int, init: str = "rand", eye_noise: float = 1e-3):
        super().__init__()
        if init == "eye":
            # Pure identity makes z_prime == z_orig exactly (and inv/ortho penalties minimal),
            # so most losses/gradients start at or near zero; add tiny noise to break symmetry.
            T = torch.eye(d) + eye_noise * torch.randn(d, d)
            T_ = torch.eye(d) + eye_noise * torch.randn(d, d)
        elif init == "rand":
            T = torch.randn(d, d) * 0.01 + torch.eye(d)
            T_ = torch.randn(d, d) * 0.01 + torch.eye(d)
        else:
            raise ValueError("init must be 'eye' or 'rand'")
        self.T = nn.Parameter(T)
        self.T_ = nn.Parameter(T_)

    def forward(self, X: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        # X: [B, T, d]
        # W: [d, d] (out, in)
        TW = self.T @ W
        sparse_term = X @ TW.T     # [B, T, d]
        z_recon = sparse_term @ self.T_ + b
        z_orig = X @ W.T + b # [B, T, d]
        return z_recon, z_orig, sparse_term

class SAE(nn.Module):
    def __init__(self, d: int, init: str = "rand", eye_noise: float = 1e-3):
        super().__init__()
        if init == "eye":
            # Pure identity makes z_prime == z_orig exactly (and inv/ortho penalties minimal),
            # so most losses/gradients start at or near zero; add tiny noise to break symmetry.
            U = torch.eye(d) + eye_noise * torch.randn(d, d)
            S = torch.eye(d) + eye_noise * torch.randn(d, d)
        elif init == "rand":
            U = torch.randn(d, d) * 0.01 + torch.eye(d)
            S = torch.randn(d, d) * 0.01 + torch.eye(d)
        else:
            raise ValueError("init must be 'eye' or 'rand'")
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(S)

    def forward(self, X: torch.Tensor):
        sparse_term = X @ self.U.T
        recon = sparse_term @ self.S.T
        return sparse_term, recon

class MLPSAE(nn.Module):
    def __init__(self, d1: int, d2: int, init: str = "rand", eye_noise: float = 1e-3):
        super().__init__()
        if init == "rand":
            U = torch.randn(d1, d1) * eye_noise + torch.eye(d1)
            S = torch.randn(d2, d1) * eye_noise + torch.eye(min(d1, d2))
        else:
            raise ValueError("init must be 'rand'")
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(S)
    
    def forward(self, X: torch.Tensor, W: torch.tensor, b: torch.tensor):
        sparse = X @ self.U.T
        recon = sparse @ self.S.T
        return sparse, recon


def token_batches(dataset_split, batch_size: int, tokenizer, device, max_length):
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

def token_batches_squad(dataset_split, batch_size: int, tokenizer, device, max_length):
    for i in range(0, len(dataset_split["input_ids"]), batch_size):
        batch = {k: torch.tensor(dataset_split[k][i : i + batch_size]).to(device) for k in dataset_split.features}
        yield batch

def _run_dev_eval(model, ds, batch_size, tokenizer, device, max_length) -> float:
    """
    Go through ds and run model on it, and save the loss it outputs
    """
    total_loss = 0.0
    n_batches = 0

    torch.manual_seed(0)

    texts = ds["text"]
    total_batches = (len(texts) + batch_size - 1) // batch_size

    total_loss, total_masked = 0.0, 0

    for batch in tqdm(
        token_batches(ds, batch_size, tokenizer, device, max_length),
        total=total_batches,
        desc="dev eval",
    ):
        tokens = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        # for sent in tokens:
        #     print()
        #     print(sent)
        labels = batch['input_ids'].clone().to(device)
        rand = torch.rand(batch['input_ids'].shape).to(device)
        mask_arr = (rand < 0.15) * (batch['input_ids'] != 101) * (batch['input_ids'] != 102) * (batch['input_ids'] != 0)
        labels[~mask_arr] = -100
        batch['input_ids'][mask_arr] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels)
            batch_loss = outputs.loss.item()
            n_batches += 1

        n_masked = mask_arr.sum().item()
        total_masked += n_masked
        total_loss += batch_loss * n_masked

    return total_loss, total_masked

def squad_f1(pred_ids, true_ids):
    # pred, true: 1D tensors (token ids)

    if type(pred_ids) != torch.Tensor:
        pred_ids = torch.tensor(pred_ids)
        true_ids = torch.tensor(true_ids)

    # get unique tokens and counts
    pred_vals, pred_counts = pred_ids.unique(return_counts=True)
    true_vals, true_counts = true_ids.unique(return_counts=True)

    # find intersection
    common = 0
    for val, p_count in zip(pred_vals, pred_counts):
        mask = (true_vals == val)
        if mask.any():
            t_count = true_counts[mask][0]
            common += min(p_count.item(), t_count.item())

    if common == 0:
        return 0.0

    precision = common / len(pred_ids)
    recall = common / len(true_ids)

    return 2 * precision * recall / (precision + recall)

def compute_metrics(eval_pred, input_data):
    predictions, labels = eval_pred
    start_preds, end_preds = predictions
    start_labels, end_labels = labels

    # start_preds = np.array(start_preds)
    # end_preds = np.array(end_preds)
    # start_labels = np.array(start_labels)
    # end_labels = np.array(end_labels)

    # lets just use HF implementation here
    #     
    f1_scores = []
    
    for i in tqdm(range(len(start_preds)), leave=False, desc="computing f1"):

        starts, ends, scores, _ = select_starts_ends(
            start_preds[i].reshape(1, -1), end_preds[i].reshape(1, -1),
            np.logical_not(input_data['mask'][i]), np.array(input_data['attention_mask'][i]),
        )

        s, e = starts[0], ends[0]
        s_t, e_t = start_labels[i], end_labels[i]
        pred_ids, true_ids = input_data['input_ids'][i][s:e+1], input_data['input_ids'][i][s_t:e_t+1]

        f1_scores.append(squad_f1(pred_ids, true_ids))

    return {'mean_f1': np.mean(f1_scores).item()}

def _run_dev_squad_eval(model, tokenizer, val_size=200, batched=True):
    # Evaluates model on SQuAD val set
    model.eval()
    dataset = load_from_disk("data/bert_cased_squad_tokenized")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "start_positions", "end_positions", "mask", "answer_mask"]
    )

    val = dataset["validation"].select(range(val_size))

    if batched:
        # run model on val set in batch mode and compute F1 score
        with torch.no_grad():
            try:
                input_ids = torch.tensor(val["input_ids"])
                attention_mask = torch.tensor(val["attention_mask"])
                start_positions = torch.tensor(val["start_positions"])
                end_positions = torch.tensor(val["end_positions"])
            except:
                input_ids = torch.stack(list(val["input_ids"]))
                attention_mask = torch.stack(list(val["attention_mask"]))
                start_positions = torch.stack(list(val["start_positions"]))
                end_positions = torch.stack(list(val["end_positions"]))

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

            eval_pred = (
                (outputs.start_logits.cpu(), outputs.end_logits.cpu()),
                (start_positions.cpu(), end_positions.cpu())
            )

            return compute_metrics(eval_pred, val)
    else:
        # run model on val set incrementally and compute F1 score
        start_logits_list = []
        end_logits_list = []
        start_positions_list = []
        end_positions_list = []
        masks = []
        snapped_source_masks = []
        snapped_input_ids = []

        with torch.no_grad():
            for i in tqdm(range(len(val)), leave=False):
                input_ids = torch.tensor(val["input_ids"][i]).unsqueeze(0)
                attention_mask = torch.tensor(val["attention_mask"][i]).unsqueeze(0)
                start_positions = torch.tensor(val["start_positions"][i]).unsqueeze(0)
                end_positions = torch.tensor(val["end_positions"][i]).unsqueeze(0)
                source_mask = torch.tensor(val["mask"][i]).unsqueeze(0)

                # Snap inputs based on padding tokens
                valid_length = attention_mask.sum().item()
                snapped_input_ids.append(val["input_ids"][i][:valid_length])
                attention_mask = attention_mask[:, :valid_length]
                source_mask = source_mask[:, :valid_length]

                outputs = model(input_ids=input_ids[:, :valid_length], attention_mask=attention_mask)

                start_logits_list.append(outputs.start_logits.cpu())
                end_logits_list.append(outputs.end_logits.cpu())
                start_positions_list.append(start_positions.cpu())
                end_positions_list.append(end_positions.cpu())
                masks.append(attention_mask.cpu())
                snapped_source_masks.append(source_mask.cpu())

        eval_pred = (
            (start_logits_list, end_logits_list),
            (start_positions_list, end_positions_list)
        )

        val_data = {
            "input_ids": snapped_input_ids,
            "attention_mask": masks,
            "mask": snapped_source_masks
        }

        return compute_metrics(eval_pred, val_data)
    
