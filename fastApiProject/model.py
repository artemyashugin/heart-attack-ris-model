import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model/Heart_Attack_Risk_Model.pkl"
THRESHOLD_PATH = BASE_DIR / "model/threshold.json"

class Model():
    # кэширование
    _model = None
    _thr = None

    def __init__(self,data):
        self.data = data
        # импорт модели
        if Model._model is None:
            try:
                Model._model = joblib.load(MODEL_PATH)
            except Exception  as e:
                raise RuntimeError(f'Cannot load model - {e}') from e
        #импорт порога
        if Model._thr is None:
            try:
                with open(THRESHOLD_PATH) as f:
                    Model._thr = json.load(f)['thr']
            except Exception  as e:
                raise RuntimeError(f'Cannot load threshold.json - {e}') from e

        self.model = Model._model
        self.thr = Model._thr

    # приведение пола к числовому типу
    def gender(self,df):
        df['Gender'] = (
            df['Gender']
            .replace({
                'Male': 0,
                'Female': 1
            })
            .astype(float)
            .astype('int64', errors='ignore')
        )
    # создание признака incomplete
    def incomplete(self,df):
        df['incomplete'] = \
            df['Diabetes'].isna() & \
            df['Smoking'].isna() & \
            df['Obesity'].isna()
    # приведение типов
    def types(self,df):
        int_cols = ['Diabetes', 'Family History', 'Smoking',
                    'Obesity', 'Alcohol Consumption',
                    'Previous Heart Problems', 'Stress Level', 'incomplete']
        df[int_cols] = df[int_cols].astype('int64', errors='ignore')
    # вызов модели
    def __call__(self,):
        df = self.data.copy()
        # приведение пола к числовому типу
        try:
            self.gender(df)
        except Exception as e:
            return {
                'status' : False,
                'error' : f'Cannot convert \'gender\' to integer - {e}'
            }
        # создание признака incomplete
        try:
            self.incomplete(df)
        except Exception as e:
            return {
                'status': False,
                'error': f'Cannot convert \'incomplete\' column - {e}'
            }
        # приведение типов
        try:
            self.types(df)
        except Exception as e:
            return {
                'status': False,
                'error': f'Cannot convert types - {e}'
            }
        #предсказание модели - возвращает JSON
        try:
            prediction = (self.model.predict_proba(df)[:, 1] >= self.thr).astype(int)
            prediction_message = {
                'id': df['id'].tolist(),
                'prediction': prediction.tolist()
            }
            return {
                'status' : True,
                'message' : prediction_message
            }
        except Exception as e:
            return {
                'status': False,
                'error': f'Cannot complete prediction - {e}'
            }


