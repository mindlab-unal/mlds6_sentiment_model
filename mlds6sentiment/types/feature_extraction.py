from pydantic import BaseModel
from typing import Optional, List

class FeatureExtractionFields(BaseModel):
    label: str
    text: str
    optionals: Optional[List[str]] = None
