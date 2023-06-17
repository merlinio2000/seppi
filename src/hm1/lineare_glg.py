from typing import Literal, Union
import numpy as np
import scipy.linalg as spl

from ..util.utl import assert_square, assert_dimensions_match, assert_eq_shape, assert_is_vec, is_diagonaldominant, bcolors
from error import apriori_n_steps_vec


def gauss(A: np.ndarray, b: np.ndarray, pretty_print: bool=True) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Gauss Verfahren für Ax=b mit optionalem print der einzelnen schritte

    Parameters:
        A: n x n Matrix
        b: Vektor in R^n
        pretty_print=True: ob die einzelnen Gauss-Schritte ausgegeben werden solllen
    Returns:
        A: obere Dreiecksmatrix welche durch gauss aus A entsteht
        det_A: die Determinante von A
        x: Die Lösung des Gleichungsystems
    """
    assert_square(A)
    assert_is_vec(b)
    assert_dimensions_match(A, b)

    def print_step(A, b, lambd, lambd_idx, pivot_idx):
        if not pretty_print:
            return
        with np.printoptions(precision=4, linewidth=150):
            matr = ' ' + str(A).replace('[', '').replace(']', '')
            vec = str(b)[1:-1]
        
        matr_lines = matr.splitlines()
        vec_lines = vec.split(' ')

        # Zeilen werden bei Gauss immer subtrahiert (nur manchmal mit negativem Faktor)
        # deshalb hier die Invertierung falls - * -
        operation = '+' if lambd < 0 else '-' 

        for i, (matr_line, vec_line) in enumerate(zip(matr_lines, vec_lines)):
            print(f'{matr_line} | {vec_line}', f'   {operation} {lambd} * {pivot_idx+1}. Zeile' if i == lambd_idx else '')

    A = A.astype(np.float64)
    b = b.astype(np.float64)
    
    
    n = A.shape[0]
    # Vorzeichen fuer det_A
    s = 1
    for i in range(n):
        if A[i,i] == 0:
            idx, = np.nonzero(A[i:,i])
            if idx.size == 0:
                raise Exception('Matrix A ist singulär')
            else:
                j = i+idx[0]
                # Zeilen tauschen
                A[[i,j], :] = A[[j,i], :]
                b[[i,j]] = b[[j,i]]
                s *= -1
         
        # Elimination
        for j in range(i+1,n):
            lambd = A[j,i]/A[i,i]
            A[j,:] -= lambd*A[i,:]
            b[j] -= lambd*b[i]
            print_step(A, b, lambd, j, i)
    
        
    # rückwaertseinsetzten
    x = np.zeros(b.shape)
    for i in range(n-1,-1,-1):
        x[i] = (b[i] - A[i,i+1:].dot(x[i+1:]))/A[i,i] 
    
    # det_A
    det_A: float = 1.
    for i in range(n):
        det_A *= A[i,i]
    det_A *= s
    
    return A, det_A, x


def RLP_zerlegung(A: np.ndarray, vertauschen: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RLP Zerlegung (optional ohne Vertauschen = RL-Zerlegung)
    Lösbar als:
    Ax = b <=> LRx = Pb
    -> Ly = Pb
    -> Rx = y

    Parameters:
        A: n x n Matrix zu zerlegen
        vertauschen = True: ob zeilenvertauschen erlaubt ist, 
                    falls False entspricht P der Einheitsmatrix
    Returns:
        R: rechts obere Dreiecksmatrix
        L: Eliminierungsfaktoren
        P: Permutationsmatrix(Zeilentausch)
    """
    assert_square(A)
 
    A = A.astype(np.float64)
    n = A.shape[0]
    L = np.zeros_like(A)
    R = A.copy() 
    P = np.eye(n, n) 

    for i in range(n - 1):
        pivot_zeile = i + np.argmax(np.abs(A[i:, i])) 
        if np.isclose(A[pivot_zeile, i], 0):
            raise Exception('Matrix ist singulär')

        if vertauschen and i != pivot_zeile:
            R[[i, pivot_zeile], :] = R[[pivot_zeile, i], :]
            L[[i, pivot_zeile], :] = L[[pivot_zeile, i], :]
            P[[i, pivot_zeile], :] = P[[pivot_zeile, i], :]


        for elim_zeile in range(i + 1, n):
            eliminations_faktor = R[elim_zeile, i] / R[i, i]
            L[elim_zeile, i] = eliminations_faktor
            R[elim_zeile, :] -= eliminations_faktor*R[i, :]

    np.fill_diagonal(L, 1)

    if not np.allclose(L @ R, P @ A):
        raise Exception('Fehler in der Zerlegung entdeckt')

    return R, L, P


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

    if not is_diagonaldominant(A):
        print(f'{bcolors.WARNING}Warnung: A ist nicht Diagonaldominant\n{bcolors.OKBLUE} -> Zeilenvertauschen kann helfen{bcolors.ENDC}')

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
        raise Exception('Kein Abbruchkriterium möglich/abstossender Fixpunkt\n'
                + 'Stattdessen muss eine fixe Anzahl Iterationen gemacht werden')

    x_prev = x0
    x_curr = step(x0)

    expected_n = apriori_n_steps_vec(tol, alpha, x_curr, x_prev)

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

    def test_FS2020_A5b(self):
        A = np.array([[240, 120, 80],
                    [60, 180, 170],
                    [60, 90, 500]], dtype=np.float64)
        b = np.array([3_080, 4_070, 5_030], dtype=np.float64)

        _, _, x = gauss(A, b, pretty_print=False)

        expected = np.array([[3, 15, 7]], dtype=np.float64)

        self.assertTrue(np.allclose(x, expected))

    def test_gauss(self):
        A = np.array([[-1, 2, 3, 2, 5, 4, 3, -1], [3, 4, 2, 1, 0, 2, 3, 8], [2, 7, 5, -1, 2, 1, 3, 5],
                      [3, 1, 2, 6, -3, 7, 2, -2], [5, 2, 0, 8, 7, 6, 1, 3], [-1, 3, 2, 3, 5, 3, 1, 4],
                      [8, 7, 3, 6, 4, 9, 7, 9], [-3, 14, -2, 1, 0, -2, 10, 5]])
        b = np.array([-11, 103, 53, -20, 95, 78, 131, -26])
        
        _, det_A, x = gauss(A, b, pretty_print=False)
        self.assertAlmostEqual(det_A, np.linalg.det(A))
        self.assertTrue(np.allclose(x, np.linalg.solve(A, b)))


    def test_RLP_SW07_BSP4_7(self):
        A = np.array([[3, 9, 12, 12],
                    [-2, -5, 7, 2],
                    [6, 12, 18, 6],
                    [3, 7, 38, 14]], dtype=np.float64)
        
        R, L, P = RLP_zerlegung(A)

        expected_R = np.array([
            [6, 12, 18, 6],
            [0, 3, 3, 9],
            [0, 0, 28, 8],
            [0, 0, 0, 3]], dtype=np.float64)
        expected_L = np.array([
            [1, 0, 0, 0],
            [0.5, 1, 0, 0],
            [0.5, 1/3, 1, 0],
            [-1/3, -1/3, 0.5, 1],
        ], dtype=np.float64)
        expected_P = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        self.assertTrue(np.allclose(expected_R, R))
        self.assertTrue(np.allclose(expected_L, L))
        self.assertTrue(np.allclose(expected_P, P))


if __name__ == '__main__':
    unittest.main()
