import numpy as np
from seppi.utl import eq_shape


def err_priori(n: int, alpha: np.float64, x1: np.ndarray, x0: np.ndarray):
   eq_shape(x1, x0)
   return ((alpha**n) / (1-alpha)) * np.linalg.norm(x1 - x0, np.inf)
    
def apriori_n_steps(tol: np.float64, alpha: np.float64, x1: np.ndarray, x0: np.ndarray):
    eq_shape(x1, x0)
    norm = np.linalg.norm(x1 - x0, np.inf)
    return np.log((tol * (1-alpha)) / norm) / np.log(alpha)

def err_posteriori(alpha: np.float64, x1: np.ndarray, x0: np.ndarray):
    eq_shape(x1, x0)
    norm = np.linalg.norm(x1 - x0, np.inf)
    return (alpha / (1 - alpha)) * norm
