import torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from vis import plot_heatmap, plot_lines

class AbsValueModel(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.encoder = nn.Linear(n, m, bias=False)
        self.decoder = nn.Linear(m, n)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.encoder(x))
        y = self.relu(self.decoder(h))
        return y

class Paper(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.decoder = nn.Linear(m, n)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.decoder.weight
        y = self.relu(self.decoder(h))
        return y

class Autoencoder(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.encoder = nn.Linear(n, m)
        self.decoder = nn.Linear(m, n)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.encoder(x))
        y = self.relu(self.decoder(h))
        return y

class PaperWithEncoder(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.encoder = nn.Linear(n, m, bias=False)
        self.decoder = nn.Linear(m, n)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        y = self.relu(self.decoder(h))
        return y

class AutoencoderNoBias(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.encoder = nn.Linear(n, m, bias=False)
        self.decoder = nn.Linear(m, n, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.encoder(x))
        y = self.relu(self.decoder(h))
        return y
    
class MaxModel(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.encoder = nn.Linear(n, m, bias=False)
        self.decoder = nn.Linear(m, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.encoder(x))
        y = self.relu(self.decoder(h))
        return y.reshape(-1, 1)

def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model_name: str,
    max_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    optimizer_class: type[optim.Optimizer] = optim.Adam,
    importance: float = 0.7
) -> None:
    """
    Train the SmallToyModel using batches and an optimizer, with early stopping.

    Args:
        model (SmallToyModel): The model to train.
        train_x (torch.Tensor): Input data of shape (N, n).
        train_y (torch.Tensor): Target data of shape (N, n).
        model_name (str): Name of the model file to save.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each batch.
        learning_rate (float): Learning rate for optimization.
        optimizer_class (type): Optimizer class (default: Adam).
        patience (int): Number of epochs to wait for improvement before stopping.
    """
    dataset = TensorDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # loss_fn = nn.MSELoss()
    def importance_mse(pred, target, I):
        loss = (pred - target)**2
        importance_vector = I ** (torch.arange(pred.shape[-1]).float())
        loss = importance_vector * loss
        return loss.sum()

    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    epochs_no_improve = 0
    model_path = os.path.join("superposition/models", f"{model_name}.pth")
    os.makedirs("superposition/models", exist_ok=True)

    progress_bar = tqdm(range(max_epochs), desc="Training", unit="epoch")

    for epoch in progress_bar:
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = importance_mse(predictions, batch_y, importance)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        progress_bar.set_postfix(loss=f"{avg_loss:.6f}")

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)  # Save best model
        else:
            epochs_no_improve += 1

# Example Usage
N, n, m, B, E = 25000, 20, 5, 32, 200  # Dataset size, input dim, hidden dim, batch size, epochs
train_x = torch.rand(N, n) * 2 + 1  # [1, 3]
I = 0.9
S = 0.99
mask = torch.bernoulli((1-S) * torch.ones_like(train_x))

train_x = mask * train_x

train_y = train_x

serialize = lambda name: f"{name}_{N}_{n}_{m}_{B}{'_' + str(S) if S > 0 else ''}"
model_name = serialize('test_abs_smallint')

model = AbsValueModel(n, m)

model_path = f'superposition/models/{model_name}.pth'

train = False
train = True

if train:
    if os.path.exists(model_path):
        train = input("Model exists, re-train? (y/n): ")
        if train.lower() == 'y':
            train_model(model, train_x, train_y, batch_size=B, model_name=model_name, max_epochs=E)
    else:
        train_model(model, train_x, train_y, batch_size=B, model_name=model_name, max_epochs=E)

model.load_state_dict(torch.load(model_path))
model.eval()

x = torch.bernoulli((1-S) * torch.ones(1, 20)) * (torch.rand(1, 20) * 2 + 1)
print(f"Input: {x}")
print(f"Output: {model(x)}")

We = model.encoder.weight if not isinstance(model, Paper) else model.decoder.weight

def feature_overlap(M: torch.tensor):
    # columns are feature embeddings here
    overlap = M.T @ M
    mask = torch.ones(*overlap.shape) - torch.eye(*overlap.shape)
    return torch.sum((overlap * mask)**2, dim = -1)

plot_heatmap([We.T @ We, feature_overlap(We).reshape(-1, 1)]) if not isinstance(model, Paper) else plot_heatmap([We @ We.T, feature_overlap(We.T).reshape(-1, 1)])