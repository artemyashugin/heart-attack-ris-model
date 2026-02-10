from fastapi import FastAPI
import uvicorn
import argparse
import logging
from model import Model
import os
from pandas import read_csv

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

app = FastAPI()

def is_csv(path: str) -> bool:
    return path.lower().endswith(".csv")

@app.post("/predict")
async def model_prediction(path : str):
    # проверка пути
    if not os.path.exists(path):
        app_logger.error('Path or file does not exist')
        return {'status': False,'error':'Path or file does not exist'}
    # проверка наличия csv
    if not path.lower().endswith(".csv"):
        app_logger.error('Only .csv files are allowed')
        return {"status": False, "error": "Only .csv files are allowed"}
    # чтение файла
    try:
        data = read_csv(path)
    except Exception as e:
        app_logger.exception("Failed to read csv: %s", path)
        return {"status": False, "error": f"Failed to read csv: {e}"}
    #предсказание
    try:
        m = Model(data)
        prediction = m()
        return prediction
    except RuntimeError as e:
        app_logger.exception("Model init failed")
        return {"status": False, "error" : str(e)}
    except Exception as e:
        app_logger.exception("Prediction crashed")
        return {"status": False, "error": f"Prediction crashed: {e}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)