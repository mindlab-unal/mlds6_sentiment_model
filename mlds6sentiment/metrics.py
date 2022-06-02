from numpy import ndarray as Array
from abc import ABC, abstractmethod
from mlds6sentiment.types.metrics import (
        MetricsKind, ClassificationMetrics,
        ClassificationMetricsEnum, Metrics
        )
from sklearn.metrics import (
        accuracy_score, f1_score,
        precision_score, recall_score
        )

class AbstractMetricsCalculator(ABC):

    metrics_kind: MetricsKind

    @abstractmethod
    def compute(self) -> Metrics:
        ...

class ClassificationMetricsCalculator(AbstractMetricsCalculator):
    metrics = {
            ClassificationMetricsEnum.ACCURACY: accuracy_score,
            ClassificationMetricsEnum.F1_SCORE: f1_score,
            ClassificationMetricsEnum.PRECISION: precision_score,
            ClassificationMetricsEnum.RECALL: recall_score
            }

    def __init__(self, metrics_kind: MetricsKind):
        self.metrics_kind = metrics_kind

    def compute(
            self, y_true: Array, y_pred: Array
            ) -> ClassificationMetrics:
        results = {}
        for metric in ClassificationMetricsEnum:
            if self.metrics_kind.dict()[metric.name.lower()]:
                results[metric.name.lower()] = self.metrics[metric](
                        y_true, y_pred
                        )
        return ClassificationMetrics(**results)
