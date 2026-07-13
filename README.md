# Weather Forecast — Temperature Prediction with PyTorch

Нейросеть предсказывает температуру в Базеле на завтра
на основе погодных данных 18 европейских городов.

## Датасет

European Climate Assessment & Dataset (ECA&D), 2000-2010,
3654 дня, 163 признака (18 городов: температура, влажность,
давление, облачность и др.)

## Архитектура

PyTorch, полносвязная сеть:
Linear(163→128) → BatchNorm → ReLU →
Linear(128→64) → BatchNorm → ReLU →
Linear(64→32) → ReLU → Linear(32→1)

## Результат

Val RMSE: ~1.63°C (Early Stopping, ~28 эпох)

## Стек

Python, PyTorch, Pandas, Scikit-learn (StandardScaler),
FastAPI, Docker

## Структура проекта

weather_forecast/
├── data/              # датасет (не в репозитории)
├── src/
│   ├── dataset.py     # загрузка и подготовка данных
│   ├── model.py        # архитектура WeatherNet
│   └── train.py        # обучение с early stopping
├── api/
│   └── main.py          # FastAPI сервис
├── Dockerfile
└── requirements.txt

## Как запустить

### Локально
1. pip install -r requirements.txt
2. Скачать датасет с Kaggle → положить в data/
3. python src/train.py — обучить модель
4. uvicorn api.main:app --reload — запустить API

### Docker
docker build -t weather-forecast .
docker run -p 8000:8000 weather-forecast

## API

POST /predict
{
  "features": [163 числа]
}
→ {"predicted_temperature": 12.5}

Документация: http://localhost:8000/docs
