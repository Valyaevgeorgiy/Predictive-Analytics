"""
This is full life cycle for ml model.
Linear regression with 5 features:
- total_meters
- floors_count
- first_floor
- last_floor
- n_rooms (One Hot Encoded)
"""

import argparse
import datetime
import glob
import os

import cianparser
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor

TEST_SIZE = 0.2
N_ROOMS = 1  # just for the parsing step
MODEL_NAME = "best_regression_v1.pkl"

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def parse_cian(n_rooms=1):
    """
    Parse data to data/raw
    :param int n_rooms: The number of flats rooms
    :return None
    """
    moscow_parser = cianparser.CianParser(location="Москва")

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f"data/raw/{n_rooms}_{t}.csv"
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 2,
            "object_type": "secondary",
        },
    )
    df = pd.DataFrame(data)

    df.to_csv(csv_path, encoding="utf-8", index=False)


def preprocess_data(test_size):
    """
    Filter, sort and remove duplicates
    """
    raw_data_path = "./data/raw"
    file_list = glob.glob(raw_data_path + "/*.csv")
    logging.info(f"Preprocess_data. Use files to train: {file_list}")
    df = pd.read_csv(file_list[0])
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i])
        df_i = pd.DataFrame(data)
        df = pd.concat([df, df_i], axis=0)

    df["url_id"] = df["url"].map(lambda x: x.split("/")[-2])
    df = (
        df[["url_id", "total_meters", "floor", "floors_count", "rooms_count", "price"]]
        .set_index("url_id")
        .sort_index()
    )

    df.drop_duplicates(inplace=True)
    df = df[df["price"] < 100_000_000]
    df = df[df["total_meters"] < 100]

    df["rooms_1"] = df["rooms_count"] == 1
    df["rooms_2"] = df["rooms_count"] == 2
    df["rooms_3"] = df["rooms_count"] == 3
    df["first_floor"] = df["floor"] == 1
    df["last_floor"] = df["floor"] == df["floors_count"]
    df.drop(columns=["floor", "rooms_count"], inplace=True)

    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)

    logging.info(f"Preprocess_data. train_df: {len(train_df)} samples")
    train_head = "\n" + str(train_df.head())
    logging.info(train_head)
    logging.info(f"Preprocess_data. test_df: {len(test_df)} samples")
    test_head = "\n" + str(test_df.head())
    logging.info(test_head)

    train_df.to_csv("data/processed/train.csv")
    test_df.to_csv("data/processed/test.csv")


def train_model(model_path):
    """Train model and save with MODEL_NAME"""
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df[
        [
            "total_meters",
            "floors_count",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor",
        ]
    ]
    y = train_df["price"]
    
    # Определение пайплайна и сетки гиперпараметров
    # Базовый pipeline
    base_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', Ridge())
    ])
    # Обёртка для лог-трансформации таргета
    ttr = TransformedTargetRegressor(
        regressor=base_pipe,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # GridSearchCV по разным моделям
    # Сетка гиперпараметров
    param_grid = [
        # Ridge
        {
            'regressor__reg': [Ridge()],
            'regressor__reg__alpha': np.logspace(-3, 3, 13)
        },
        # ElasticNet
        {
            'regressor__reg': [ElasticNet(max_iter=10000, random_state=42)],
            'regressor__reg__alpha': np.logspace(-4, 1, 10),
            'regressor__reg__l1_ratio': [0.1, 0.5, 0.9]
        },
        # RandomForest
        {
            'regressor__reg': [RandomForestRegressor(random_state=42)],
            'regressor__reg__n_estimators': [100, 200],
            'regressor__reg__max_depth': [None, 10, 20]
        },
        # ExtraTrees
        {
            'regressor__reg': [ExtraTreesRegressor(random_state=42)],
            'regressor__reg__n_estimators': [100, 200],
            'regressor__reg__max_depth': [None, 10, 20]
        },
        # GradientBoosting
        {
            'regressor__reg': [GradientBoostingRegressor(random_state=42)],
            'regressor__reg__n_estimators': [100, 200],
            'regressor__reg__learning_rate': [0.01, 0.1],
            'regressor__reg__max_depth': [3, 5]
        },
        # HistGradientBoosting
        {
            'regressor__reg': [HistGradientBoostingRegressor(random_state=42)],
            'regressor__reg__max_iter': [100, 200],
            'regressor__reg__max_depth': [None, 10, 20]
        }
    ]
    
    model = GridSearchCV(
        estimator=ttr,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    # Обучаем
    model.fit(X, y)
    best = model.best_estimator_
    logging.info(f"Best params: {model.best_params_}")
    logging.info(f"Best CV R2: {model.best_score_:.3f}")

    logging.info(f"Train {model} and save to {model_path}")

    joblib.dump(best, model_path)


def test_model(model_path):
    """Test model with new data"""
    test_df = pd.read_csv("data/processed/test.csv")
    train_df = pd.read_csv("data/processed/train.csv")
    X_test = test_df[
        [
            "total_meters",
            "floors_count",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor",
        ]
    ]
    y_test = test_df["price"]
    X_train = train_df[
        [
            "total_meters",
            "floors_count",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor",
        ]
    ]
    y_train = train_df["price"]
    model = joblib.load(model_path)
    
    # Предсказание на тестовой выборке
    y_test_pred = model.predict(X_test)

    # Оценка модели
    mse_te = mean_squared_error(y_test, y_test_pred)
    rmse_te = np.sqrt(mse_te)
    mae_te = mean_absolute_error(y_test, y_test_pred)
    r2_te = r2_score(y_test, y_test_pred)
    logging.info(f"Test MSE: {mse_te:.2f}")
    logging.info(f"Test RMSE: {rmse_te:.2f}")
    logging.info(f"Test MAE: {mae_te:.2f}")
    logging.info(f"Test R2: {r2_te:.3f}")


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test size, from 0 to 0.5",
        default=TEST_SIZE,
    )
    parser.add_argument(
        "-n", "--n_rooms", help="Number of rooms to parse", type=int, default=N_ROOMS
    )
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    parser.add_argument(
        "-p", "--parse_data", help="Flag to parse new data", action="store_true"
    )
    args = parser.parse_args()

    test_size = float(args.split)
    assert 0.0 <= test_size <= 0.5
    model_path = os.path.join("models", args.model)

    if args.parse_data:
        parse_cian(args.n_rooms)
    preprocess_data(test_size)
    train_model(model_path)
    test_model(model_path)