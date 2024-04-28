import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Definición de la clase ConvertidorDummies
class ConvertidorDummies(BaseEstimator, TransformerMixin):
    def __init__(self, variables_cualitativas):
        self.variables_cualitativas = variables_cualitativas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.get_dummies(X, columns=self.variables_cualitativas, drop_first=True)
