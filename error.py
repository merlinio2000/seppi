import numpy as np
from utl import assert_eq_shape


def err_priori(n: int, alpha: float, x1: float, x0: float):
   delta = abs(x1 - x0)
   return ((alpha**n) / (1-alpha)) * delta 

def apriori_n_steps(tol: float, alpha: float, x1: float, x0: float):
    delta = abs(x1 - x0)
    return np.log((tol * (1-alpha)) / delta) / np.log(alpha)

def err_posteriori(alpha: float, x_curr: float, x_prev: float):
    delta = abs(x_curr - x_prev)
    return (alpha / (1 - alpha)) * delta


def err_priori_vec(n: int, alpha: float, x1: np.ndarray, x0: np.ndarray):
   assert_eq_shape(x1, x0)
   return ((alpha**n) / (1-alpha)) * np.linalg.norm(x1 - x0, np.inf)
    
def apriori_n_steps_vec(tol: float, alpha: float, x1: np.ndarray, x0: np.ndarray):
    assert_eq_shape(x1, x0)
    norm = np.linalg.norm(x1 - x0, np.inf)
    return np.log((tol * (1-alpha)) / norm) / np.log(alpha)

def err_posteriori_vec(alpha: float, x_curr: np.ndarray, x_prev: np.ndarray):
    assert_eq_shape(x_curr, x_prev)
    norm = np.linalg.norm(x_curr - x_prev, np.inf)
    return (alpha / (1 - alpha)) * norm
