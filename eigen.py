from dataclasses import dataclass
import numpy as np
from seppi.utl import is_symmetric

from utl import assert_dimensions_match, assert_normed, assert_square


def eigen_QR(A: np.ndarray, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """
    QR-Zerlegung zur Bestimmung der Eigenwerte

    Parameters:
        A: n x n Matrix deren Eigenwerte bestimmt werden sollen
        max_iter: Anzahl Iterationen
    Returns:
        A_i: Entspricht B/D; Matrix die die EW auf der Diagonale hat (oder 2x2 Blöcke für komplexe EW)
        P_i: Entspricht T falls A_i symmetrisch ist und alle EW betragsmässig verschieden sind
    """
    assert_square(A)

    A_curr = A.astype(np.float64)
    n = A.shape[0]

    P_curr = np.eye(n)

    for i in range(max_iter):
        Q_curr, R_curr = np.linalg.qr(A_curr)
        A_curr = R_curr @ Q_curr
        P_curr = P_curr @ Q_curr

    return (A_curr, P_curr)


@dataclass(frozen=True)
class EigenQRAnalysisResult:
    """
    Infos zum resultat einer Eigenwertbestimmung mit QR

    Attributes:
        A_symmetrisch: ob die Eingabematrix A symmetrisch_ist
        B_ist_D: Ob die Matrix B = T^-1 A T diagonal ist 
                (in diesem Fall spricht mann in Formeln von D = B)
        EV_sind_spalten_von_T: kann nur True sein falls #matrix_ist_diagonal || #A_symmetrisch = True
                Falls True können Eigenvektoren aus Spalten von T abgelesen werden
                Dies ist der Fall wenn A diagonalisierbar (== #B_ist_D)
                oder A symmetrisch ist und alle Elemente der Diagonale **betragsmässig** verschieden sind 
    """
    A_symmetrisch: bool = False
    B_ist_D: bool = False
    EV_sind_spalten_von_T: bool = False


def eigen_QR_analysis(A, max_iter=100) -> EigenQRAnalysisResult:
    assert_square(A)

    A_symmetrisch = is_symmetric(A)

    B, T = eigen_QR(A, max_iter)

    n = B.shape[0]
    
    # Test für komplexe Eigenwerte
    is_diag_matrix = True
    for i in range(n):
        for k in range(n):
            if i != k:
                is_diag_matrix = np.isclose(B[i, k], 0)

    if not is_diag_matrix: # TODO correct
        print("Matrix scheint komplexe Eigenwerte zu haben, manuell prüfen")
        return
    D = B # für konsistenz mit mathematischer Schreibweise
    print("Eigenwerte stehen auf der Diagonalen von Folgender Matrix")
    print(D)

    # Test ob alle Eigenwerte betragsmässig verschieden sind
    seen_ews = []
    all_absolute_different = True
    for ew in np.diag(D):
        abs_ew = np.abs(ew)
        for seen_ew in seen_ews:
            if np.isclose(abs_ew, seen_ew):
                all_absolute_different = False
                break
        if not all_absolute_different:
            break

    if all_absolute_different and A_symmetrisch:
        print("Eigenvektoren stehen in den Spalten folgender Matrix T")
        print(T)
        return(A_symmetrisch, B_ist_D)
    else:
        print("Eigenvektoren müssen manuell bestimmt werden")
    




def von_mises_iter(A: np.ndarray, v_0: np.ndarray, max_iter: int) -> tuple[np.ndarray, float]:
    """
    von-Mises Iteration für einen gegebenen Startvektor v_0

    Parameters:
        A: die n x n Matrix deren EW & EV bestimmt werden soll
        v_0: Startvektor für die Iteration
        max_iter: maximal erlaubte Anzahl iteration
    Returns:
        v: Eigenvektor nach Iteration
        lambd: zugehöriger Eigenwert nach der Iteration
    """
    assert_square(A)
    assert_dimensions_match(A, v_0)
    assert_normed(v_0)

    n = A.shape[0]

    v_curr = v_0.reshape(n, 1).astype(np.float64)

    lambd: float = np.nan

    for k in range(max_iter):
        v_curr = (A @ v_curr) / np.linalg.norm(A @ v_curr, 2)
        lambd = ((v_curr.T @ A @ v_curr) / (v_curr.T @ v_curr))[0] # TODO verify

    return v_curr, lambd



import unittest

class EigenTest(unittest.TestCase):
    def test_Skript_BSP_4_24(self):
        A = np.array([[1, -2, 0],
                    [2, 0, 1],
                    [0, -2, 1]], dtype=np.float64)
        D, _ = eigen_QR(A, 100)

        expected = np.array([[-0.1072, -2.5543, -0.6586],
                        [2.3955, 1.1072, -0.2575],
                        [0, 0, 1]])
        self.assertTrue(np.allclose(D, expected, atol=1e-04))

    def test_HS2020_A6(self):
        A = np.array([[13, -4],
                    [30, -9]], dtype=np.float64)
        D, _ = eigen_QR(A, 100)
        eigen_values

if __name__ == '__main__':
    unittest.main()