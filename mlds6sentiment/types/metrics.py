from pydantic import BaseModel, PositiveFloat
from typing import Optional
from enum import Enum, auto

class MetricsKind(BaseModel):
    accuracy: Optional[bool] = False
    f1_score: Optional[bool] = False
    precision: Optional[bool] = False
    recall: Optional[bool] = False

Metrics = BaseModel

class ClassificationMetricsEnum(Enum):
    ACCURACY = auto()
    F1_SCORE = auto()
    PRECISION = auto()
    RECALL = auto()

class ClassificationMetrics(Metrics):
    accuracy: Optional[PositiveFloat] = None
    f1_score: Optional[PositiveFloat] = None
    precision: Optional[PositiveFloat] = None
    recall: Optional[PositiveFloat] = None
