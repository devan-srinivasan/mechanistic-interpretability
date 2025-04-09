# model_and_training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from tqdm import tqdm
from vis import plot_heatmap

# ---- Config ----
VOCAB = list(range(1, 10))  # [1..9]
VOCAB_SIZE = len(VOCAB) + 1  # +1 for <s>
IDX_TO_TOKEN = ['<s>'] + [str(i) for i in VOCAB]
TOKEN_TO_IDX = {tok: i for i, tok in enumerate(IDX_TO_TOKEN)}

SEQ_LEN = 6
EMBED_DIM = 12
NUM_HEADS = 1
NUM_LAYERS = 2
EPOCHS = 5000

class OneHotEmbedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        return F.one_hot(x, num_classes=self.vocab_size).float()

class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, D = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Hd)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, Hd)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        return self.out_proj(attn_output)

class SimpleMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super().__init__()
        self.attn = SimpleSelfAttention(embed_dim, num_heads)
        self.mlp = SimpleMLP(embed_dim, mlp_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = OneHotEmbedding(VOCAB_SIZE)
        self.proj = nn.Linear(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.randn(SEQ_LEN, EMBED_DIM))
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerBlock(EMBED_DIM, NUM_HEADS, mlp_hidden_dim=128)
            for _ in range(NUM_LAYERS)
        ])
        self.output = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        x = self.proj(self.token_embed(x))  # (B, T, D)
        x = x + self.pos_embed.unsqueeze(0)  # add positional encoding
        for block in self.transformer_blocks:
            x = block(x)
        return self.output(x)  # (B, T, vocab_size)

# ---- Dataset ----
def generate_sequence():
    prefix = random.sample(VOCAB, 5)
    idx = random.randint(0, 3)  # choose an earlier number
    query = prefix[idx]
    target = prefix[idx + 1]
    seq = prefix + [query]
    return seq, target

def collate_batch(batch):
    xs, ys = zip(*batch)
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

# ---- Training ----
def train_model():
    model = TinyTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    os.makedirs("models", exist_ok=True)
    model_path = f"induction_heads/models/{SEQ_LEN}_{EMBED_DIM}_{NUM_HEADS}_{NUM_LAYERS}.pth"
    pbar = tqdm(total=EPOCHS, desc="Training", unit='epoch')
    for _ in range(EPOCHS):
        batch = [generate_sequence() for _ in range(32)]
        x, y = collate_batch(batch)
        logits = model(x)  # (B, T, V)
        last_logits = logits[:, -1, :]  # (B, V)
        loss = loss_fn(last_logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update(1)
        pbar.set_postfix({'loss': loss.item()})
        if loss.item() < 0.005:
            break

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# ---- Evaluation ----
def load_model():
    model = TinyTransformer()
    model_path = f"induction_heads/models/{SEQ_LEN}_{EMBED_DIM}_{NUM_HEADS}_{NUM_LAYERS}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def pred(model: nn.Module, input_seq: list, target: int):
    x = torch.tensor([input_seq], dtype=torch.long)
    logits = model(x)
    pred = torch.argmax(logits[:, -1, :], dim=-1).item()
    return IDX_TO_TOKEN[pred], IDX_TO_TOKEN[target], pred == target

def test_model(n=100):
    model = load_model()
    with torch.no_grad():
        num_correct = 0
        for _ in range(n):
            seq, target = generate_sequence()
            predicted, target, correct = pred(model, seq, target)
            num_correct += int(correct)
            # print(f"{seq} => {predicted}{'('+target+')' if target != pred else ''}")
    print(f"{num_correct}/{n}")

# if __name__ == "__main__":
#     train_model()
#     test_model(1000)

model = load_model()
block1, block2 = model.transformer_blocks

# first layer attention
print(model)