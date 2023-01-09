from typing import Literal, Union
import numpy as np
import scipy.linalg as spl

from utl import assert_square, assert_dimensions_match, assert_eq_shape, assert_is_vec
from error import apriori_n_steps


def np_RLP_zerlegung(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zur überprüfung ob die RLP Zerlegung richtig gemacht wurde
    Zum überprüfen der RL Zerlegung muss die P Matrix der Einheitsmatrix entsprechen

    Parameters:
        A: n x n Matrix zu zerlegen
    Returns:
        R: obere Dreiecksmatrix
        L: untere Dreiecksmatrix (Enthält die Einzelnen Faktoren zur Eliminierung aus Gauss)
        P: Transformationsmatrix
    """
    P, L, R = spl.lu(A) # type: ignore
    return R, L, P


def QR_zerlegung(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    QR Zerlegung für eine n x n Matrix A, so dass QR = A

    Parameters:
        A: die Matrix
    Returns:
        Q: orthogonale Matrix
        R: obere Dreiecksmatrix
    """
    assert_square(A)
    
    A = A.astype(np.float64)
    
    n = np.shape(A)[0]
    
    
    Q = np.eye(n)
    R = A
    
    for j in np.arange(0,n-1):
        a = np.copy(R[:,j][j:]).reshape(n-j,1) # spaltenvektor des neuen R's
        e = np.eye(n-j, 1) # spalte der einheitsmatrix
        length_a = np.linalg.norm(a, 2) # länge des spalten vektors
        if a[0] >= 0: sig = 1
        else: sig = -1
        v = a + (sig * length_a * e)
        u = v/np.linalg.norm(v)
        H = np.eye(n-j) - (2 * u @ u.T)
        Qi = np.eye(n)
        Qi[j:,j:] = H
        R = Qi @ R
        Q = Q @ Qi.T
        
    return (Q,R)


def gaussSeidel_or_jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray, \
                    tol: float = 1e-9, \
                    opt: Union[Literal['g'], Literal['j']] = 'g') -> tuple[np.ndarray, int, float]:
    """
    Gauss-Seidel oder Jacobi verfahren für das System Ax = b, DEFAULT = Gauss-Seidel

    Parameters:
        A: n x n Matrix
        b: Vektor aus Gleichungssystem
        x0: Startvektor für die iteration
        tol = 1e-9: Fehlertoleranz für die Lösung
        opt = 'g': Modus; 'g' / 'j' respektive für Gauss-Seidel/Jacobi
    Returns:
        x_n: Lösungsvektor x nach n Iterationen
        n: Anzahl der durchgeführten Iterationen
        expected_n: Anzahl der erwarteten Iterationen aus Formel
    """    

    assert_square(A)
    assert_is_vec(b)
    assert_is_vec(x0)
    assert_dimensions_match(A, b)
    assert_eq_shape(b, x0)

    A = A.astype(np.float64)
    b = b.astype(np.float64)
    x0 = x0.astype(np.float64)    

    R = np.triu(A, 1)
    L = np.tril(A, -1)
    D = np.diagflat(np.diag(A))
    
    B = None
    C = None
    
    if (opt == 'j'): # Jacobi
        D_inv = np.linalg.inv(D)
        B = -D_inv @ (L + R)
        C = D_inv @ b
    elif (opt == 'g'): # Gauss-Seidel
        DL_inv = np.linalg.inv(D + L)
        B = -DL_inv @ R
        C = DL_inv @ b
    else:
        raise Exception(f'Unknown mode {opt}')
    
    def step(xk): return B @ xk + C

    alpha: float = np.linalg.norm(B, np.inf) # type: ignore
    if (alpha >= 1):
        raise Exception('Kein Abbruchkriterium möglich/abstossender Fixpunkt')

    x_prev = x0
    x_curr = step(x0)

    expected_n = apriori_n_steps(tol, alpha, x_curr, x_prev)

    err_fact = alpha / (1 - alpha)
    
    n_steps = 1 # wir haben bereits eine Iteration um expected_n zu berechnen können
    while err_fact * np.linalg.norm(x_curr - x_prev, np.inf) > tol:
        x_prev = x_curr
        x_curr = step(x_curr)
        n_steps += 1
    
    return x_curr, n_steps, expected_n 







import unittest

class LineareGlgTest(unittest.TestCase):
    def test_QR_S8_A2(self):
        A = np.array([[1, -2, 3],
                    [-5, 4, 1],
                    [2, -1, 3]], dtype=np.float64)
        
        myQ, myR = QR_zerlegung(A)

        npQ, npR = np.linalg.qr(A)

        self.assertTrue(np.allclose(myQ, npQ))
        self.assertTrue(np.allclose(myR, npR))

    def test_RL_ohne_vertauschen(self):
        A = np.array([[20, 30, 10],
                    [10, 17, 6],
                    [2, 3, 2]], dtype=np.float64)
        A = A * 1e3

        _, _, P = np_RLP_zerlegung(A)
        self.assertTrue(np.allclose(P, np.eye(3)))
    
    def test_gaussSeidel_S10_A2a(self):
        A = np.array([[8, 5, 2],
                [5, 9, 1],
                [4, 2, 7]], dtype=np.float64)

        b = np.array([19, 5, 34], dtype=np.float64)

        x0 = np.array([1, -1, 3])

        x_n, _, expected_n  = gaussSeidel_or_jacobi(A, b, x0, 1e-4)
        expected_x = np.array([2.00000527, -1.00000223, 3.99999763])

        self.assertTrue(np.allclose(x_n, expected_x))
        self.assertAlmostEqual(expected_n, 86.22, places=2)



if __name__ == '__main__':
    unittest.main()