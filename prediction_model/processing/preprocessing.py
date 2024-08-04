
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MeanImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables = None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mean_dict_ = {}

        for col in self.variables:
            self.mean_dict_[col] = X[col].mean()

        return self
    
    def transform(self,X):
        X = X.copy()

        for col in self.variables:
           X[col] = X[col].fillna(self.mean_dict_[col])

        return X

class ModeImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables = None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.mode_dict_ = {}

        for col in self.variables:
            self.mode_dict_[col] = X[col].mode()[0]

        return self
    
    def transform(self,X):
        X = X.copy()

        for col in self.variables:
          X[col] = X[col].fillna(self.mode_dict_[col])

        return X

class DropColumns(BaseEstimator,TransformerMixin):
    def __init__(self,drop_cols = None):
        self.drop_cols = drop_cols
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        return X.drop(columns = self.drop_cols)
    

class DomainProcessing(BaseEstimator,TransformerMixin):
    def __init__(self,var_to_modify = None,var_to_add= None):
        self.var_to_modify = var_to_modify
        self.var_to_add = var_to_add
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for ft in self.var_to_modify:
            X[ft] = X[ft] + X[self.var_to_add]
        return X
    

class CategoryEncoder(BaseEstimator,TransformerMixin):
    def __init__(self, variables = None):
        self.variables = variables
    
    def fit(self,X,y=None):
        self.label_dict_ = {}
        
        for cat in self.variables:
            t = X[cat].value_counts().sort_values(ascending=True).index
            self.label_dict_[cat] = {k:i for i,k in enumerate(t,0)}

        return self
    
    def transform(self,X):
        X = X.copy()
        for cat in self.variables:
            X[cat] = X[cat].map(self.label_dict_[cat])
        return X
    
    
class LogTransform(BaseEstimator,TransformerMixin):
    def __init__(self, variables = None):
        self.variables = variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.log(X[var])
        return X
    