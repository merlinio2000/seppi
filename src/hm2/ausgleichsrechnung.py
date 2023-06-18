import numpy as np
import src.util.types as types
import src.util.utl as utl


def fehlerfunktional(x: types.NPValue, y_exakt: types.NPValue, f_ansatz: types.NPValueToValueFn) \
        -> float:
    '''
    Bestimmt das Fehlerfunktional einer Ansatzfunktion f am Punkt x, f(x) nach
    der Methode der kleinsten Fehlerquadrate

    Parameters:
        x: x an dem das Fehlerfunktional bestimmt wird
        y_exakt: das EXAKTE y zu diesem x
        f_ansatz: die Ansatzfunktion f
    Returns:
        float: das Fehlerfunktional E(f) = ||y - f(x)||_2^2
    '''
    return np.linalg.norm(y_exakt - f_ansatz(x), 2)**2 # type: ignore


def lin_ausgleich(x: np.ndarray, y: np.ndarray, fs: list[types.NPValueToValueFn]) \
        -> np.ndarray:
    '''
    Bestimmt lambdas l_i der linearen Ausgleichsfunktion f(x) = l_1 * f_1(x) + 
        ... + l_m * f_m(x)
    sodass das das Fehlerfunktional der kleinsten Fehlerquadrate minimal wird.

    Parameters:
        x: x der Messpunkte, länge = n
        y: y der Messpunkte, länge = n
        fs: liste der Basisfunktionen f_1,...,f_m; länge = m
    Returns:
        Vektor: mit einzelnen Lambdas l_i
    '''
    utl.assert_is_vec(x)
    utl.assert_eq_shape(x, y)

    fs_evaluated = [f(x) for f in fs]

    base_len = len(fs_evaluated[0]) # type: ignore
    assert all([isinstance(row, np.ndarray) and len(row) == base_len \
            for row in fs_evaluated])

    A = np.column_stack(fs_evaluated)

    print('linearer Ausgleich')
    print('A-Matrix:')
    utl.np_pprint(A)
   
    # Besser konditioniert als A^T A
    Q, R = np.linalg.qr(A)
    
    return np.linalg.solve(R, Q.T @ y)


def nichtlin_ausgleich():
    raise Exception('todo')




import unittest

class AusgleichsTest(unittest.TestCase):
    def test_lin_ausgleich_S6_A1(self):
        x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64)
        y = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4], dtype=np.float64)
        assert len(x) == len(y)

        f1 = lambda x: x**2
        f2 = lambda x: x
        f3 = lambda x: np.ones_like(x)
        fs = [f1, f2, f3]

        def f(x, lams):
            return np.sum([lam*f_i(x) for lam, f_i in zip(lams, fs)], axis=0)

        our_lambdas = lin_ausgleich(x, y, fs)

        np_lambdas = np.polyfit(x, y, 2)

        self.assertTrue(np.allclose(our_lambdas, np_lambdas))
        
        # import matplotlib.pyplot as plt
        #
        # x_int = np.linspace(0, 100, num=100_000)
        # y_np = np.polyval(np_lambdas, x_int)
        # y_ours_f = f(x_int, our_lambdas)
        #
        # plt.plot(x, y, 'bx', label='Messpunkte')
        # plt.plot(x_int, y_ours_f, 'r', label='unser')
        # plt.plot(x_int, y_np, 'g', label='numpy')
        # plt.legend()
        # plt.grid()
        # plt.show()
        


