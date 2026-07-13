import torch
import torch.nn as nn


class WeatherNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    model = WeatherNet(input_size=163)
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nПараметров: {total}")

    # Тест forward pass
    x = torch.randn(32, 163)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")