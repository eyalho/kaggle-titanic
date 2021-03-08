import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class BaselineV1Transformer(BaseEstimator, TransformerMixin):
    """ Most Naive model I came with"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if "Survived" in X.columns:
            X = X.drop("Survived", axis=1)  # remove label

        # For baseline: random forest with numeric values only
        X.Sex = X.Sex.map({"male": 1, "female": -1})
        X.Embarked = X.Embarked.map({"C85": 1, "C123": 2, "B42": 3})
        X = X.drop(["Name", "Ticket", "Cabin"], axis=1)
        X = X.fillna(0)
        return X


class BaselineV2Transformer(BaseEstimator, TransformerMixin):
    """ Use SimpleImputer instead of fillna"""

    def __init__(self):
        self.my_simple_imputer = SimpleImputer()

    def fit(self, X, y=None):
        X = self._transform_without_imputer(X)
        self.my_simple_imputer.fit(X)
        return self

    def transform(self, X, y=None):
        X = self._transform_without_imputer(X)
        X = self.my_simple_imputer.transform(X)
        return X

    def _transform_without_imputer(self, X):
        X = X.copy()
        if "Survived" in X.columns:
            X = X.drop("Survived", axis=1)  # remove label

        # For baseline: random forest with numeric values only
        X.Sex = X.Sex.map({"male": 1, "female": -1})
        X.Embarked = X.Embarked.map({"C85": 1, "C123": 2, "B42": 3})
        X = X.drop(["Name", "Ticket", "Cabin"], axis=1)
        return X


class BaselineV3Transformer(BaseEstimator, TransformerMixin):
    """ Create another boolean features named "miss_featrue" and use SimpleImputer"""

    def __init__(self):
        self.my_was_missing_imputer = SimpleImputer()
        self.fitted_data = None

    def fit(self, X, y=None):
        X = self._transform_without_imputer(X)
        self.my_was_missing_imputer.fit(X)
        self.fitted_data = X
        return self

    def transform(self, X, y=None):
        X = self._transform_without_imputer(X)
        _, X = self.fitted_data.align(X, join="left", axis=1)

        X = self.my_was_missing_imputer.transform(X)
        return X

    def _transform_without_imputer(self, X):
        X = X.copy()
        if "Survived" in X.columns:
            X = X.drop("Survived", axis=1)  # remove label

        # For baseline: random forest with numeric values only
        X.Sex = X.Sex.map({"male": 1, "female": -1})
        X.Embarked = X.Embarked.map({"C85": 1, "C123": 2, "B42": 3})
        X = X.drop(["Name", "Ticket", "Cabin"], axis=1)

        cols_with_missing = (col for col in X.columns if X[col].isnull().any())
        for col in cols_with_missing:
            X[col + '_was_missing'] = X[col].isnull()

        return X


class BaselineV4Transformer(BaseEstimator, TransformerMixin):
    """ Use get_dummy -> create another boolean featrues to handle catogrical data instead of map """

    def __init__(self):
        self.my_was_missing_imputer = SimpleImputer()
        self.fitted_data = None

    def fit(self, X, y=None):
        X = self._transform_without_imputer(X)
        self.my_was_missing_imputer.fit(X)
        self.fitted_data = X
        return self

    def transform(self, X, y=None):
        X = self._transform_without_imputer(X)
        _, X = self.fitted_data.align(X, join="left", axis=1)

        X = self.my_was_missing_imputer.transform(X)
        return X

    def _transform_without_imputer(self, X):
        X = X.copy()
        if "Survived" in X.columns:
            X = X.drop("Survived", axis=1)  # remove label

        X = X.drop(["Name", "Ticket", "Cabin"], axis=1)
        X = pd.get_dummies(X)

        cols_with_missing = (col for col in X.columns if X[col].isnull().any())
        for col in cols_with_missing:
            X[col + '_was_missing'] = X[col].isnull()

        return X
