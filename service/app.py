from flask import Flask, render_template, request 
from logging.config import dictConfig
import joblib

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },         
            "file": {
                "class": "logging.FileHandler",
                "filename": "service/flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)

# Сохранение модели
# model_path = 'models/linear_regression_model.pkl'

# loaded_model = joblib.load(model_path)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    # Здесь можно добавить обработку полученных чисел
    # Для примера просто возвращаем их обратно
    data = request.get_json()
    
    app.logger.info(f'Requst data: {data}')
    
    try:
        price = float(data['area']) * 300_000
    except ValueError:
        return {'status': 'error', 'data': 'Ошибка парсинга данных'}
    
    return {'status': 'success', 'data': price}

if __name__ == '__main__':
    app.run(debug=True)