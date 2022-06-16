# The Custom transformers inherit from the sklearn.base classes 
# (BaseEstimator, TransformerMixin). This makes them compatible with 
# scikit-learn’s Pipelines
from sklearn.base import BaseEstimator, TransformerMixin

from mfccUtilities import *
from melUtilities import *
import numpy as np
import random


class Trimmer(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        # return the trimmed audio files
        for idx in range(len(X)):
            X[idx] = trim(X[idx])
        return X

class LengthNormalizer(BaseEstimator, TransformerMixin):
    # initializer
    def __init__(self, max_ms: int, random: bool) -> None:
        self.max_ms = max_ms
        self.random = random

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        res = []
        for idx in range(len(X)):
            sig, sr = X[idx]
            # Using librosa.load we have mono = True then num_rows = 1
            num_rows = 1
            sig_len = len(sig)
            max_len = sr//1000 * self.max_ms

            if (sig_len > max_len):
                # Truncate the signal to the given length
                sig = sig[:max_len]

            elif (sig_len < max_len):
                # Length of padding to add at the beginning and end of the signal
                if self.random:
                    pad_begin_len = random.randint(0, max_len - sig_len)
                    pad_end_len = max_len - sig_len - pad_begin_len
                    pad_begin = np.zeros((num_rows, pad_begin_len))
                    pad_end = np.zeros((num_rows, pad_end_len))
                    sig = np.concatenate([pad_begin, sig, pad_end])
                else:
                    pad_end_len = max_len - sig_len
                    # Pad with 0s
                    pad_end = np.zeros((num_rows, pad_end_len))
                    sig = np.concatenate([sig, pad_end])

            res.append((sig, sr))
        return res

class Mfcc(BaseEstimator, TransformerMixin):
    # initializer
    def __init__(self, n_mfcc: int = 12) -> None:
        self.n_mfcc = n_mfcc

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        res = []
        # Compute the mfcc
        for idx in range(len(X)):
            res.append(compute_mfcc(X[idx]))
        return res

class MfccAverageCalculator(BaseEstimator, TransformerMixin):
    # initializer
    def __init__(self) -> None:
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        res = []
        # Compute the mfcc mean
        for idx in range(len(X)):
            res.append(compute_mfcc_mean(X[idx]))
        return np.array(res)

class MfccStdCalculator(BaseEstimator, TransformerMixin):
    # initializer
    def __init__(self) -> None:
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        res = []
        # Compute the mfcc std
        for idx in range(len(X)):
            res.append(compute_mfcc_std(X[idx]))
        return np.array(res)
