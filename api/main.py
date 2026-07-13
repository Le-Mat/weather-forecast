from pydantic import BaseModel
from typing import List
import numpy as np
from fastapi import FastAPI
import torch
import joblib
import os
from src.model import WeatherNet

class WeatherInput(BaseModel):
    features: List[float]


app = FastAPI()

# Пути к файлам — от корня проекта
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'model.pth')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

# Загружаем scaler
scaler = joblib.load(scaler_path)

# Загружаем модель
model = WeatherNet(input_size=163)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: WeatherInput):
    # Превращаем список в numpy array, форма (1, 163) — один объект
    X = np.array(input_data.features).reshape(1, -1)

    # Масштабируем через ТОТ ЖЕ scaler что использовался при обучении
    X_scaled = scaler.transform(X)

    # В тензор
    X_tensor = torch.FloatTensor(X_scaled)

    # Предсказание
    with torch.no_grad():
        prediction = model(X_tensor)

    return {"predicted_temperature": prediction.item()}