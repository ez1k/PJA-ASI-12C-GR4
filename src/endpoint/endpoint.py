import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import joblib
from pydantic import BaseModel
import pandas as pd

class InputJson(BaseModel):
    Company: int
    TypeName: int
    Inches: float
    Cpu: int
    Ram: int
    Gpu: int
    Weight: float
    IsFullHD: int
    IsIPS: int
    IsTouchscreen: int
    IsRetina: int
    ResolutionWidth: int
    ResolutionHeight: int
    IsHDD: int
    IsSSD: int
    IsFlashStorage: int
    FlashStorage: float
    MemorySizeHDD_TB: float
    MemorySizeHDD_GB: float
    MemorySizeSSD: float
    CpuSpeed: float
    CpuBrand: int
    GpuBrand: int
    OperatingSystem: int

app = FastAPI()

@app.post('/')
def predict(request: InputJson):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../data/04_model/best_model.pickle')
    lr = joblib.load(path)

    df = pd.DataFrame(request.__dict__, index=[0])
    prediction = lr.predict(df)
    print(prediction)
    return jsonable_encoder({'prediction': list(prediction)})
