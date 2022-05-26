from pydantic import BaseModel
from mlds6sentiment.types.models import ModelKind, SentimentsClassifier
from mlds6sentiment.types.hparams import (
        SVMHParams, RandomForestHParams, LogisticHParams
        )
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ModelProxy:
    model_table = {
            ModelKind.SVM: [SVC, SVMHParams],
            ModelKind.LOGISTIC: [LogisticRegression, LogisticHParams],
            ModelKind.RANDOM_FOREST: [RandomForestClassifier, RandomForestHParams]
            }
    def __init__(self) -> None:
        self.model_kind: ModelKind = ModelKind.SVM
        self.hparams: BaseModel = SVMHParams()

    def add_model_kind(self, model_kind: ModelKind) -> "ModelProxy":
        self.model_kind = model_kind
        return self

    def add_hparams(self, *args, **kwargs) -> "ModelProxy":
        self.hparams = self.model_table[self.model_kind][1](*args, **kwargs)
        return self

    def resolve(self) -> SentimentsClassifier:
        model_class, _ = self.model_table[self.model_kind]
        model = model_class(**self.hparams.dict())
        return model
