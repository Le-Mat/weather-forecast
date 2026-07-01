import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def explore_data(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape}")
    print(f"Период: {df['DATE'].min()} — {df['DATE'].max()}")
    print(f"Пропуски: {df.isna().sum().sum()}")
    print(f"\nСтатистика по целевой переменной (BASEL_temp_mean):")
    print(df['BASEL_temp_mean'].describe())


def prepare_data(df: pd.DataFrame):
    # Убираем DATE и MONTH — не нужны как признаки
    feature_cols = [col for col in df.columns
                    if col not in ['DATE', 'MONTH']]

    # Целевая переменная — температура в Базеле завтра
    # Сдвигаем на 1 день вперёд
    target = df['BASEL_temp_mean'].shift(-1)

    # Убираем последнюю строку (у неё нет завтрашней температуры)
    X = df[feature_cols].iloc[:-1].values.astype(np.float32)
    y = target.iloc[:-1].values.astype(np.float32)

    return X, y


class WeatherDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = load_data(os.path.join(base_dir, 'data', 'weather_prediction_dataset.csv'))
    explore_data(df)

    X, y = prepare_data(df)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Пример y (первые 5): {y[:5]}")