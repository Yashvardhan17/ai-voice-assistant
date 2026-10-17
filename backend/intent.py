from pathlib import Path
from functools import lru_cache
import joblib
ROOT=Path(__file__).resolve().parents[1]
@lru_cache(maxsize=1)
def model():
    path=ROOT/'artifacts'/'intent_model.joblib'
    if not path.exists(): raise RuntimeError('Model missing. Run: python -m ml.train')
    return joblib.load(path)
def predict(text:str):
    clf=model()
    probabilities=clf.predict_proba([text])[0]
    best=int(probabilities.argmax())
    return {'intent':str(clf.classes_[best]),'confidence':round(float(probabilities[best]),4)}
