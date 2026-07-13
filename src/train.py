import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_data, prepare_data, WeatherDataset
from model import WeatherNet
from math import inf

import joblib

def train(n_epochs=100, batch_size=32, lr=0.0005):
    # --- Данные ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = load_data(os.path.join(base_dir, 'data', 'weather_prediction_dataset.csv'))
    X, y = prepare_data(df)

    # Масштабирование
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    # Train/Val split (80/20)
    dataset = WeatherDataset(X, y)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Модель ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    model = WeatherNet(input_size=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Обучение ---
    train_losses = []
    val_losses = []

    # --- Итоги обучения ---
    best_val_loss = inf
    patience = 15
    no_improve = 0
    best_model_state = None

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            model_path = os.path.join(base_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if patience == no_improve: break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Val loss: {val_loss:.4f} | "
                  f"Val RMSE: {val_loss**0.5:.2f}°C")

    model.load_state_dict(best_model_state)
    model_path = os.path.join(base_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nЛучшая модель сохранена: {model_path}")

    # --- График ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Weather Forecast Training')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'training_plot.png'))
    plt.show()
    print("График сохранён: training_plot.png")

    return model, scaler


if __name__ == '__main__':
    model, scaler = train(n_epochs=100, batch_size=32, lr=0.001)