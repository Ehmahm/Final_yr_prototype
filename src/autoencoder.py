import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, seq_len, hidden=64, latent=16):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, n_features, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        h_dec = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder(h_dec)
        return out, z


def train_autoencoder(sequences, seq_len, n_features, epochs=20, batch_size=64, lr=1e-3):
    """
    Trains an LSTM autoencoder on a set of sequences.
    Returns the trained model and the latent vectors for all sequences.
    """
    X = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(n_features, seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            out, _ = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.6f}")

    model.eval()
    with torch.no_grad():
        _, latents = model(X)
    return model, latents.numpy()


if __name__ == "__main__":
    from src.data import load_raw_data, resample_hourly, resample_daily, build_sequences

    raw = load_raw_data()
    hourly = resample_hourly(raw)
    daily = resample_daily(hourly)

    feature_cols = hourly.columns.tolist()

    # Daily-scale autoencoder (24h windows)
    daily_seqs = build_sequences(hourly, feature_cols, seq_len=24)
    print("Daily sequences shape:", daily_seqs.shape)
    daily_model, daily_latents = train_autoencoder(daily_seqs, seq_len=24, n_features=len(feature_cols))
    print("Daily latents shape:", daily_latents.shape)

    # Monthly-scale autoencoder (30-day windows)
    monthly_seqs = build_sequences(daily, feature_cols, seq_len=30)
    print("Monthly sequences shape:", monthly_seqs.shape)
    monthly_model, monthly_latents = train_autoencoder(monthly_seqs, seq_len=30, n_features=len(feature_cols))
    print("Monthly latents shape:", monthly_latents.shape)