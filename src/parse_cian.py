"""Parse data from cian.ru
https://github.com/lenarsaitov/cianparser
"""

import datetime
import logging
import argparse
import cianparser
import pandas as pd

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

N_ROOMS = 1
moscow_parser = cianparser.CianParser(location="Москва")

def parse_cian(n_rooms, n_pages):
    """
    Function docstring
    """
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f"data/raw/{n_rooms}_{t}.csv"

    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": n_pages,
            "object_type": "secondary"
        },
    )
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, encoding="utf-8", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n_rooms", help="Number of rooms to parse", type=int, default=N_ROOMS
    )
    parser.add_argument(
        "-p", "--n_pages", help="Number of pages to parse", type=int, default=2
    )
    args = parser.parse_args()
    n_rooms = args.n_rooms
    n_pages = args.n_pages
    
    assert n_rooms in (1, 2, 3), "Number of rooms invalid"
    assert 2 <= n_pages <= 20 , "Number of pages invalid"

    parse_cian(n_rooms, n_pages)