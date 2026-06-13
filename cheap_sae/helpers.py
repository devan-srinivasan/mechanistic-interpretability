from tqdm import tqdm
import torch, torch.nn as nn

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
        if init == "eye":
            # Pure identity makes z_prime == z_orig exactly (and inv/ortho penalties minimal),
            # so most losses/gradients start at or near zero; add tiny noise to break symmetry.
            U = torch.eye(d1) + eye_noise * torch.randn(d1, d1)
            S = torch.eye(d2) + eye_noise * torch.randn(d2, d2)
        elif init == "rand":
            U = torch.randn(d1, d1) * 0.01 + torch.eye(d1)
            S = torch.randn(d2, d2) * 0.01 + torch.eye(d2)
        else:
            raise ValueError("init must be 'eye' or 'rand'")
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(S)

    def forward_1(self, X: torch.Tensor):
        return X @ self.U.T
    
    def forward_2(self, X: torch.Tensor, W: torch.tensor, b: torch.tensor):
        return (X @ W.T + b) @ self.S.T


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