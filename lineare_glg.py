import numpy as np
import scipy.linalg as spl

from utl import assert_square


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


import unittest

class LineareGlgTest(unittest.TestCase):
    def test_S8_A2(self):
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


if __name__ == '__main__':
    unittest.main()