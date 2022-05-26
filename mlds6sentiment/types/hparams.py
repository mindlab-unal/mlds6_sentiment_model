from pydantic import BaseModel
from typing import Optional

class SVMHParams(BaseModel):
    C: float = 1.0
    kernel: str = "linear"
    gamma: Optional[float] = None

class LogisticHParams(BaseModel):
    C: float = 1.0

class RandomForestHParams(BaseModel):
    n_estimators: int = 100
    max_depth: int = 7

