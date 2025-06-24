# Metal Price Rising ML

Компактный инструмент машинного обучения для
прогнозирования цен металлов (или любых других торговых инструментов с
историческими котировками).

## Быстрый старт

### 1. Установка
```bash
git clone https://github.com/r0manb/metal-price-rising-ml.git
cd metal-price-rising-ml

pip install -r requirements.txt
```

### 2. Сбор данных
```python
from parsers.example_parser import ExampleParser # здесь должна быть ваша реализация

# Инициализируем парсер
parser = ExampleParser(ticker="GOLD", interval="1d")
```

### 3. Обучение модели
```python
from model_trainer import ModelTrainer

# Инициализируем тренер
trainer = ModelTrainer(
    data_parser=parser, # Парсер данных для обучения
    epochs=100, # Количество эпох
    window_size=7, # Размер окна для обучения
    forecast_horizon=3 # Горизонт прогноза
)

trainer.train() # Обучаем модель
trainer.save("models/gold_lstm") # Сохраняем модель
```

### 4. Инференс
```python
from predictor import Predictor

# Инициализируем предиктор
predictor = Predictor(path="models/gold_lstm", timestamp_strategy="adaptive")

# Получаем данные
raw_data = parser.fetch_data()

# Предсказываем цены на основе полученных данных
prices, future_dates = predictor.predict_with_dates(raw_data)
for ts, price in zip(future_dates, prices):
    print(ts, price)
```

## Структура репозитория
```
MPR/
├── parsers/              # парсеры данных
│   ├── data_parser.py    # абстрактный класс-шаблон
│   ├── example_parser.py # демонстрационный парсер
├── utils/                # утилиты
├── model_trainer.py      # класс трейнера
├── predictor.py          # класс инференса
├── requirements.txt      # зависимости
└── README.md
```

## Настройка модели
Все ключевые параметры ( `window_size`, `forecast_horizon` )
сохраняются в `config.json` внутри директории модели. Это позволяет загружать
модель без ручного указания гиперпараметров.

## Метрики качества
При оценке используются:
* **MAE**  — средняя абсолютная ошибка
* **MSE**  — среднеквадратичная ошибка
* **RMSE** — корень из MSE
* **MAPE** — средняя абсолютная процентная ошибка

## Требования к окружению
* Python 3.10+
* См. версии библиотек в `requirements.txt`  
  (TensorFlow 2.18, scikit-learn 1.6, pandas 2.2 и др.)

## Лицензия
Проект распространяется под лицензией MIT. См. файл `LICENSE`.