from datetime import datetime, timedelta
import os, sys
import pandas as pd
import sqlite3

from airflow import DAG
from airflow.operators.python import PythonOperator

# Путь до вашего репо
REPO_DIR = os.getenv('AIRFLOW__CORE__AIRFLOW_HOME', '/opt/airflow')
RAW_DIR = os.path.join(REPO_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(REPO_DIR, 'data', 'processed')
DB_PATH = os.path.join(REPO_DIR, 'data', 'realestate_valyaev.db')

default_args = {
    'owner': 'valyaev',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 17),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'predictive_analytics',
    default_args=default_args,
    description='Ежедневный pipeline: парсинг, загрузка в БД, обучение модели',
    schedule_interval='0 0 * * *',  # каждый день в полночь
    catchup=False,
)


def parse_room(n_rooms, **context):
    """Вызов вашей функции parse_cian из lifecycle.py"""
    # кладём /opt/airflow/src в путь импорта
    sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
    from lifecycle import parse_cian
    parse_cian(n_rooms)
    # ожидаем, что CSV ложится в data/raw

def load_to_sqlite(**context):
    """Собрать все свежие raw CSV и загрузить в SQLite"""
    con = sqlite3.connect(DB_PATH)
    for fn in os.listdir(RAW_DIR):
        if fn.startswith('1_') and fn.endswith('.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, fn), index_col=0)
            # имя таблицы с фамилией
            df.to_sql('apartments_valyaev', con, if_exists='append', index=True)
    con.close()

def preprocess_and_train(**context):
    """Запустить preprocess_data и train_model из lifecycle.py"""
    # кладём /opt/airflow/src в путь импорта
    sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
    from lifecycle import preprocess_data, train_model
    # 20% тест
    preprocess_data(test_size=0.2)
    model_path = os.path.join(REPO_DIR, 'models', 'best_regression_v1.pkl')
    train_model(model_path)


# Создаём задачи
parse_1room = PythonOperator(
    task_id='parse_1_room',
    python_callable=parse_room,
    op_kwargs={'n_rooms': 1},
    dag=dag,
)

parse_2room = PythonOperator(
    task_id='parse_2_rooms',
    python_callable=parse_room,
    op_kwargs={'n_rooms': 2},
    dag=dag,
)

parse_3room = PythonOperator(
    task_id='parse_3_rooms',
    python_callable=parse_room,
    op_kwargs={'n_rooms': 3},
    dag=dag,
)

load_db = PythonOperator(
    task_id='load_to_sqlite',
    python_callable=load_to_sqlite,
    dag=dag,
)

train = PythonOperator(
    task_id='preprocess_and_train',
    python_callable=preprocess_and_train,
    dag=dag,
)

# Описываем порядок выполнения
# [parse_1room, parse_2room, parse_3room] >> load_db >> train