import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# === Sparse Autoencoder ===
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, sparsity_target=0.05, sparsity_weight=1e-3):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        z = torch.sigmoid(self.encoder(x))  # sigmoid to bound activation in [0,1]
        x_hat = self.decoder(z)
        return x_hat, z

    def kl_divergence(self, z):
        # Compute average activation of hidden neurons over the batch
        rho_hat = torch.mean(z, dim=0)
        rho = self.sparsity_target
        # KL divergence for each neuron
        kl = rho * torch.log(rho / (rho_hat + 1e-10)) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-10))
        return torch.sum(kl)

# === Dummy training loop ===
def train_autoencoder(model, data_loader, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            x = batch.to(device)
            x_hat, z = model(x)

            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(x_hat, x)

            # Sparsity penalty
            kl_loss = model.kl_divergence(z)
            loss = recon_loss + model.sparsity_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# === Example usage ===
if __name__ == "__main__":
    # Fake 768-dim data for demo (e.g. MPNet embeddings)
    dummy_embeddings = torch.randn(1000, 768)  # 1000 samples

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(dummy_embeddings), batch_size=32, shuffle=True)

    model = SparseAutoencoder()
    train_autoencoder(model, loader)
