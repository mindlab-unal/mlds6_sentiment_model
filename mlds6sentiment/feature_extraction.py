import numpy as np
from numpy import ndarray as Array
from pandas import DataFrame
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from mlds6sentiment.types.feature_extraction import FeatureExtractionFields
from typing import Tuple

class AbstractFeatureExtractor(ABC):
    data: DataFrame
    fields: FeatureExtractionFields
    x: Array
    y: Array

    def add_data(self, data: DataFrame) -> "AbstractFeatureExtractor":
        self.data = data
        return self

    def add_fields(self, fields: FeatureExtractionFields) -> "AbstractFeatureExtractor":
        self.fields = fields
        return self

    def save(self, path: str):
        np.save(f"{path}_x.npy", self.x)
        np.save(f"{path}_y.npy", self.y)

    @abstractmethod
    def extract(self) -> Tuple[Array, Array]:
        ...

class VidhyaSentimentsFeatureExtractor(AbstractFeatureExtractor):

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def extract(self) -> Tuple[Array, Array]:
        text = self.data[self.fields.text].to_list()
        self.y = self.data[self.fields.label].to_numpy()
        self.x = self.vectorizer.fit_transform(text).toarray()
        return self.x, self.y

