import os, math, random, torch, torch.nn as nn, torch.optim as optim
from tqdm import trange

# Settings
d = 128
N_training = 10000
epochs = 10000
positions = list(range(1, 7))  # Positions from 1 to 6
noise_std = 0.25  # standard deviation of added Gaussian noise

# Output path
os.makedirs("induction_heads/positional_models", exist_ok=True)
save_path = f"induction_heads/positional_models/{d}_{N_training}_{epochs}_{str(noise_std).replace('.', '-')}.pth"

# Positional encoding function (sinusoidal, like in transformers)
def get_positional_encoding(p, d):
    pe = torch.zeros(d)
    for i in range(0, d, 2):
        div_term = 10000 ** (i / d)
        pe[i] = math.sin(p / div_term)
        if i + 1 < d:
            pe[i + 1] = math.cos(p / div_term)
    return pe

# Model: one linear layer with no bias
class PEExtractor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.linear(x)

# Load model
def load_model():
    model = PEExtractor(d)
    filename = f"induction_heads/positional_models/{d}_{N_training}_{epochs}_{str(noise_std).replace('.', '-')}.pth"
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

# Evaluation
def test_model_on_position(model, p):
    with torch.no_grad():
        clean_encoding = get_positional_encoding(p, d)
        noisy = clean_encoding + torch.randn(d) * noise_std
        recovered = model(noisy)
        return clean_encoding, recovered

def train_model():
    # Generate training data
    X = []
    Y = []
    for _ in range(N_training):
        p = random.choice(positions)
        pe = get_positional_encoding(p, d)
        noisy_input = pe + noise_std * torch.randn(d)
        X.append(noisy_input)
        Y.append(pe)  # Supervised target is the clean PE

    X = torch.stack(X)  # shape (N, d)
    Y = torch.stack(Y)

    model = PEExtractor(d)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training loop with tqdm
    pbar = trange(epochs, desc="Training", dynamic_ncols=True)
    latest_loss = None

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            latest_loss = loss.item()
            pbar.set_postfix({"loss": f"{latest_loss:.6f}"})

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# train_model()
model = load_model()
t, p = test_model_on_position(model, 4)
print(torch.abs(t-p).sum())