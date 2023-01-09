from dataclasses import dataclass
import math
import numpy as np
from utl import assert_dimensions_match, assert_normed, assert_square, is_symmetric


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

        komplexe_EW: Ob die Matrix B komplexe Eigenwerte in 2x2 Blöcken enthält

        EV_sind_spalten_von_T: kann nur True sein falls #matrix_ist_diagonal || #A_symmetrisch = True
                Falls True können Eigenvektoren aus Spalten von T abgelesen werden
                Dies ist der Fall wenn A diagonalisierbar (== #B_ist_D)
                oder A symmetrisch ist und alle Elemente der Diagonale **betragsmässig** verschieden sind 
    """
    A_symmetrisch: bool | None = None 
    B_ist_D: bool | None = None
    komplexe_EW: bool | None = None
    EV_sind_spalten_von_T: bool | None = None


def eigen_QR_analysis(A: np.ndarray, max_iter: int = 100, tol: float = 1e-9) -> EigenQRAnalysisResult:
    assert_square(A)

    np.set_printoptions(linewidth=150)

    A_symmetrisch = is_symmetric(A)

    # np_ews[i] -> np_evs[:, i] für zugehörigen EV
    np_ews, np_evs = np.linalg.eig(A)
    print("Eigenwerte nach numpy:")
    print(np_ews)
    print("normierte Eigenvektoren nach numpy: (Eigenwertindex entspricht Spaltenindex des zugehörigen EV)")
    print(np_evs)

    B, T = eigen_QR(A, max_iter)

    n = B.shape[0]
    
    # Test ob B diagonal (D = B) 
    is_diag_matrix = True
    for col in range(n):
        for row in range(n):
            if col != row:
                is_diag_matrix = math.isclose(B[col, row], 0, abs_tol=tol)
            if not is_diag_matrix:
                break
        if not is_diag_matrix:
            break

    if is_diag_matrix:
        print("Eigenwerte stehen auf der Diagonalen von folgender Matrix D = B")
        print(B)
        print("unabhängige Eigenvektoren stehen in den Spalten folgender Matrix T")
        print(T)
        return EigenQRAnalysisResult(A_symmetrisch=A_symmetrisch, \
                                        B_ist_D=True, \
                                        komplexe_EW=False, \
                                        EV_sind_spalten_von_T=True)




    # Test ob keine != 0 unter der Diagonale ("perfekte" obere Dreiecksmatrix)
    # Falls dem nicht so ist, treten komplexe Eigenwerte in 2x2 Blöcken auf
    zero_below_diagonal = True
    for col in range(n):
        for row in range(col+1, n):
            zero_below_diagonal = math.isclose(B[row, col], 0, abs_tol=tol)
            if not zero_below_diagonal:
                break
        if not zero_below_diagonal:
            break
    
    if not zero_below_diagonal:
        print("In folgender Matrix treten komplexe Eigenwerte als 2x2 Matrizen auf, manuell prüfen")
        print(B)
        return EigenQRAnalysisResult(A_symmetrisch=A_symmetrisch, \
                                        B_ist_D=False, \
                                        komplexe_EW=True, \
                                        EV_sind_spalten_von_T=False)
    else:
        print("Eigenwerte stehen auf der Diagonalen von folgender Matrix B")
        print(B)


    # Test ob alle Eigenwerte betragsmässig verschieden sind
    # Zu diesem Zeitpunkt is bereit geprüft, dass
    #   keine komplexen EW (B ist obere Dreiecksmatrix)
    #   B ist nicht diagonal
    seen_abs_ews = []
    all_absolute_different = True
    for ew in np.diag(B):
        abs_ew = np.abs(ew)
        all_absolute_different = not any(math.isclose(seen_abs_ew, abs_ew, abs_tol=tol) for seen_abs_ew in seen_abs_ews)
        seen_abs_ews.append(abs_ew)
        if not all_absolute_different:
            break

    if all_absolute_different and A_symmetrisch:
        print("unabhängige Eigenvektoren stehen in den Spalten folgender Matrix T")
        print(T)
        return EigenQRAnalysisResult(A_symmetrisch=True, \
                                        B_ist_D=False, \
                                        komplexe_EW=False, \
                                        EV_sind_spalten_von_T=True)
    else:
        print("Eigenvektoren müssen manuell bestimmt werden")
        return EigenQRAnalysisResult(A_symmetrisch=A_symmetrisch, \
                                        B_ist_D=False, \
                                        komplexe_EW=False, \
                                        EV_sind_spalten_von_T=False)
    




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
        B, _ = eigen_QR(A, 100)
        analysis = eigen_QR_analysis(A)

        expected = np.array([[-0.1072, -2.5543, -0.6586],
                        [2.3955, 1.1072, -0.2575],
                        [0, 0, 1]])
        self.assertTrue(np.allclose(B, expected, atol=1e-04))


        self.assertFalse(analysis.A_symmetrisch)
        self.assertFalse(analysis.B_ist_D)
        self.assertTrue(analysis.komplexe_EW)
        self.assertFalse(analysis.EV_sind_spalten_von_T)

    def test_HS2020_A6(self):
        A = np.array([[13, -4],
                    [30, -9]], dtype=np.float64)
        B, _ = eigen_QR(A, 100)
        analysis = eigen_QR_analysis(A)

        self.assertAlmostEqual(B[0, 0], 3.)
        self.assertAlmostEqual(B[1, 1], 1.)

        self.assertFalse(analysis.A_symmetrisch)
        self.assertFalse(analysis.B_ist_D)
        self.assertFalse(analysis.komplexe_EW)
        self.assertFalse(analysis.EV_sind_spalten_von_T)
        

if __name__ == '__main__':
    unittest.main()