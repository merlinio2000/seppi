import numpy as np
from ..util.utl import assert_eq_shape, assert_is_vec, assert_square, assert_dimensions_match


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



"""
Funktionen für die Fehler aus linearen Gleichungsystemen Ax=b
"""

def err_abs_b_fehler_A_exakt(A: np.ndarray, b: np.ndarray, b_err: np.ndarray) -> float:
    """
    Für den Fall das in der Gleichung Ax=b NUR b fehlerhaft ist
    """
    assert_square(A)
    assert_is_vec(b)
    assert_is_vec(b_err)
    assert_eq_shape(b, b_err)
    assert_dimensions_match(A, b)

    return np.linalg.norm(np.linalg.inv(A), np.inf) * np.linalg.norm(b - b_err, np.inf) # type: ignore

def err_rel_b_fehler_A_exakt(A: np.ndarray, b: np.ndarray, b_err: np.ndarray) -> float:
    """
    Für den Fall das in der Gleichung Ax=b NUR b fehlerhaft ist
    """
    assert_square(A)
    assert_is_vec(b)
    assert_is_vec(b_err)
    assert_eq_shape(b, b_err)
    assert_dimensions_match(A, b)

    if np.linalg.norm(b, np.inf) == 0:
        raise Exception('Division durch norm = 0 nicht erlaubt')
    cond = np.linalg.cond(A, np.inf)
    return cond * (np.linalg.norm(b-b_err, np.inf) / np.linalg.norm(b, np.inf))


def err_rel_b_fehler_A_fehler(A: np.ndarray, A_err: np.ndarray,  b: np.ndarray, b_err: np.ndarray) -> float:
    """
    Für den Fall das in der Gleichung Ax=b sowohl A als auch B fehlerhaft ist
    """
    assert_square(A)
    assert_eq_shape(A, A_err)
    assert_is_vec(b)
    assert_is_vec(b_err)
    assert_eq_shape(b, b_err)
    assert_dimensions_match(A, b)


    if np.linalg.norm(b, np.inf) == 0 or np.linalg.norm(A, np.inf): 
        raise Exception('Division durch norm = 0 nicht erlaubt')

    cond = np.linalg.cond(A)

    err_fact_A = np.linalg.norm(A-A_err, np.inf) / np.linalg.norm(A, np.inf)

    if cond * err_fact_A >= 1:
        raise Exception('Formel für diese Matrix A nicht anwendbar')

    err_fact_b = np.linalg.norm(b-b_err, np.inf)/np.linalg.norm(b, np.inf)

    return cond/(1-(cond * err_fact_A)) * (err_fact_A + err_fact_b) 
