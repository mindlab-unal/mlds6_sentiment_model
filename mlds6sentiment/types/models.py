from typing import Protocol, Optional
from numpy import ndarray as Array
from enum import Enum, auto

class SentimentsClassifier(Protocol):

    def fit(self, X: Array, y: Optional[Array]) -> "SentimentsClassifier":
        ...

    def predict(self, X: Array) -> Array:
        ...

class ModelKind(Enum):
    SVM = auto()
    LOGISTIC = auto()
    RANDOM_FOREST = auto()

